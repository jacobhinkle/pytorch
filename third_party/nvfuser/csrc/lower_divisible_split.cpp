
#include <lower_divisible_split.h>

#include <disjoint_set.h>
#include <ir_utils.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::unordered_set<Split*> getAllDivisibleSplits(Fusion* fusion) {
  ComputeAtMap ca_map(fusion);
  return getAllDivisibleSplits(fusion, &ca_map);
}

std::unordered_set<Split*> getAllDivisibleSplits(
    Fusion* fusion,
    const ComputeAtMap* ca_map) {
  std::unordered_set<Split*> all_divisible_splits;

  auto all_tvs = ir_utils::allTvs(fusion);
  // Find all tensor views with a view like rfactor. Splits used in view
  // transformations must be divisible by definition.
  for (auto tv : all_tvs) {
    auto rfactor_dom = tv->getMaybeRFactorDomain();
    // Not view if there's no rfactor axis
    if (!tv->domain()->hasViewLikeRFactor()) {
      continue;
    }

    // Take the view transformations and add all the splits. Those splits are
    // the only divisible splits.
    auto view_exprs =
        StmtSort::getExprs(fusion, {rfactor_dom.begin(), rfactor_dom.end()});
    auto split_exprs = ir_utils::filterByType<Split>(view_exprs);
    all_divisible_splits.insert(split_exprs.begin(), split_exprs.end());
  }

  // Vectorized dimensions are enforced to be a result of divisible splits.
  // Gather vectorized splits.
  for (auto tv : all_tvs) {
    auto vec_id_it = std::find_if(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        [](IterDomain* id) {
          return id->getParallelType() == ParallelType::Vectorize;
        });

    if (vec_id_it == tv->domain()->domain().end()) {
      continue;
    }

    // We could have a case technically like:
    // [8, 2] where we do:
    // split(0, 2)
    // merge(1)
    // so it ends up as [4, 4]
    // split(0, 2) must be divisible, but for now we're not going to capture
    // cases like this. Just look for direct split's producing a vectorize
    // dimension.
    auto vec_id = *vec_id_it;
    if (vec_id->definition() != nullptr && vec_id->definition()->isA<Split>()) {
      all_divisible_splits.emplace(vec_id->definition()->as<Split>());
    }
  }

  // If there's no view like splits, there's nothing to find
  if (all_divisible_splits.empty()) {
    return all_divisible_splits;
  }

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<Expr*>>>
      all_mapped_disjoint_expr_sets;

  for (auto divisible_split : all_divisible_splits) {
    auto set_pair = ca_map->idGraph().getDisjointExprSet(
        divisible_split, IdMappingMode::ALMOSTEXACT);
    if (set_pair.second) {
      all_mapped_disjoint_expr_sets.pushBack(set_pair.first);
    }
  }

  for (auto set : all_mapped_disjoint_expr_sets) {
    auto split_exprs = ir_utils::filterByType<Split>(set->vector());
    for (auto split_expr : split_exprs) {
      all_divisible_splits.emplace(split_expr);
    }
  }

  return all_divisible_splits;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
