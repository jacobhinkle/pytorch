#include <compute_at_map.h>

#include <disjoint_set.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <tuple>
#include <typeinfo>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace {

// Is the provided IterDomain an Leaf of provided TensorView and within its
// computeAtPosition.
// If outside computeAt axis, we don't want to directly map consumer/producer in
// the loop mapping as they are not sharing the same loop.
bool idIsAComputeAtLeafDomain(
    IterDomain* id,
    TensorView* producer_tv,
    TensorView* consumer_tv) {
  auto begin = producer_tv->domain()->domain().begin();
  auto end = producer_tv->domain()->domain().begin() +
      producer_tv->getComputePosition(consumer_tv);
  return std::find(begin, end, id) != end;
}

// Is the provided IterDomain an Leaf of provided TensorView
bool idIsALeafDomain(IterDomain* id, TensorView* tv) {
  auto begin = tv->domain()->domain().begin();
  auto end = tv->domain()->domain().end();
  return std::find(begin, end, id) != end;
}

} // namespace

IterDomainGraph::IterDomainGraph(Fusion* fusion, bool allow_self_mapping) {
  // Initialize the required sets as if a permissive relationship is never
  // found, then querying an empty permissive map will fail later.
  std::vector<IdMappingMode> mapping_types{
      IdMappingMode::EXACT,
      IdMappingMode::ALMOSTEXACT,
      IdMappingMode::PERMISSIVE,
      IdMappingMode::LOOP};

  // Initialize disjoint sets
  for (auto mode : mapping_types) {
    disjoint_ids_[mode] = DisjointSets<IterDomain*>();
    disjoint_exprs_[mode] = DisjointSets<Expr*>();
  }

  build(fusion);

  if (!allow_self_mapping) {
    TORCH_INTERNAL_ASSERT(
        !hasSelfMapping(),
        "Unsupported domain mapping detected in ",
        std::get<0>(*self_mapping_info_)->toString(),
        ". ",
        std::get<3>(*self_mapping_info_),
        " domains, ",
        std::get<1>(*self_mapping_info_)->toString(),
        " and ",
        std::get<2>(*self_mapping_info_)->toString(),
        ", are mapped with each other.");
  }
}

const DisjointSets<IterDomain*>& IterDomainGraph::getDisjointIdsSet(
    IdMappingMode mode) const {
  auto disjoint_ids_it = disjoint_ids_.find(mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_ids_it != disjoint_ids_.end(),
      "Mapping mode ",
      mode,
      " not supported.");
  return disjoint_ids_it->second;
}

DisjointSets<IterDomain*>& IterDomainGraph::disjointIdsSet(IdMappingMode mode) {
  auto disjoint_ids_it = disjoint_ids_.find(mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_ids_it != disjoint_ids_.end(),
      "Mapping mode ",
      mode,
      " not supported.");
  return disjoint_ids_it->second;
}

const DisjointSets<Expr*>& IterDomainGraph::getDisjointExprsSet(
    IdMappingMode mode) const {
  auto disjoint_exprs_it = disjoint_exprs_.find(mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_exprs_it != disjoint_exprs_.end(),
      "Mapping mode ",
      mode,
      " not supported.");
  return disjoint_exprs_it->second;
}

DisjointSets<Expr*>& IterDomainGraph::disjointExprsSet(IdMappingMode mode) {
  auto disjoint_exprs_it = disjoint_exprs_.find(mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_exprs_it != disjoint_exprs_.end(),
      "Mapping mode ",
      mode,
      " not supported.");
  return disjoint_exprs_it->second;
}

bool IterDomainGraph::exprsMap(
    Expr* first,
    Expr* second,
    bool forward,
    IdMappingMode mode) const {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (typeid(*first) != typeid(*second)) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      first->isA<Merge>() || first->isA<Split>() || first->isA<Swizzle2D>(),
      "Merge and split are the only expressions supported through rfactor operations in compute at map, but found:\n",
      first->toString());

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->inputs() : first->outputs())
                       .vector();

  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->inputs() : second->outputs())
                        .vector();

  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  {
    std::vector<std::pair<IterDomain*, IterDomain*>> zipped_ids;

    std::transform(
        first_ids.begin(),
        first_ids.end(),
        second_ids.begin(),
        std::back_inserter(zipped_ids),
        [](IterDomain* first, IterDomain* second) {
          return std::make_pair(first, second);
        });

    if (std::any_of(
            zipped_ids.begin(),
            zipped_ids.end(),
            [&](std::pair<IterDomain*, IterDomain*> id_pair) {
              return !getDisjointIdsSet(mode).permissiveAreMapped(
                  id_pair.first, id_pair.second);
            })) {
      return false;
    }
  }

  if (first->isA<Merge>() && !forward) {
    // Can't back prop through merge without making sure one dimension actually
    // is identical extents.
    auto merge0 = first->as<Merge>();
    auto merge1 = second->as<Merge>();

    auto extent_0o = merge0->outer()->extent();
    auto extent_0i = merge0->inner()->extent();
    auto extent_1o = merge1->outer()->extent();
    auto extent_1i = merge1->inner()->extent();

    auto extent_0_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluateInt() == extent_1o->evaluateInt());

    auto extent_1_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluateInt() == extent_1i->evaluateInt());

    if (!(extent_0_match || extent_1_match)) {
      return false;
    }
  }

  if (first->isA<Split>()) {
    auto first_split = first->as<Split>();
    auto second_split = second->as<Split>();
    if (!first_split->factor()->sameAs(second_split->factor()) ||
        first_split->innerSplit() != second_split->innerSplit() ||
        !first_split->startOffset()->sameAs(second_split->startOffset()) ||
        !first_split->stopOffset()->sameAs(second_split->stopOffset())) {
      return false;
    }
  }

  if (first->isA<Swizzle2D>()) {
    auto first_swizzle = first->as<Swizzle2D>();
    auto second_swizzle = second->as<Swizzle2D>();
    if (first_swizzle->swizzleMode() != second_swizzle->swizzleMode() ||
        first_swizzle->swizzleType() != second_swizzle->swizzleType()) {
      return false;
    }
  }

  return true;
}

void IterDomainGraph::mapIds(
    IterDomain* id0,
    IterDomain* id1,
    IdMappingMode mode) {
  if (mode == IdMappingMode::LOOP) {
    disjointIdsSet(mode).mapEntries(id0, id1);
    return;
  }

  if (disjointIdsSet(mode).strictAreMapped(id0, id1)) {
    return;
  }

  // Definitions and uses are based on the groups of id0 and id1, don't merge
  // them into a single group until we grab all definitions and uses for later
  // processing.

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<Expr*>>> defs0;
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<Expr*>>> defs1;
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<Expr*>>> uses0;
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<Expr*>>> uses1;

  auto group0 = disjointIdsSet(mode).disjointSetMap().at(id0);
  auto group1 = disjointIdsSet(mode).disjointSetMap().at(id1);

  if (unique_definitions_[mode].find(group0) !=
      unique_definitions_[mode].end()) {
    defs0 = unique_definitions_[mode].at(group0);
    unique_definitions_[mode].erase(group0);
  }

  if (unique_definitions_[mode].find(group1) !=
      unique_definitions_[mode].end()) {
    defs1 = unique_definitions_[mode].at(group1);
    unique_definitions_[mode].erase(group1);
  }

  if (unique_uses_[mode].find(group0) != unique_uses_[mode].end()) {
    uses0 = unique_uses_[mode].at(group0);
    unique_uses_[mode].erase(group0);
  }

  if (unique_uses_[mode].find(group1) != unique_uses_[mode].end()) {
    uses1 = unique_uses_[mode].at(group1);
    unique_uses_[mode].erase(group1);
  }

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointIdsSet(mode).mapEntries(id0, id1);

  auto id_set = disjointIdsSet(mode).disjointSetMap().at(id0);

  // Record which expression to propagate across. We want to update the
  // defintion and use maps before we propagating through other expressions.
  std::vector<std::tuple<Expr*, Expr*, bool>> expr_prop;

  // Propagate on definitions
  if (defs0.size() > 0 || defs1.size() > 0) {
    if (defs0.size() > 0 && defs1.size() > 0) {
      auto new_def_group = defs0;
      new_def_group.insert(defs1.begin(), defs1.end());

      for (auto def_group_1 : defs1) {
        if (defs0.has(def_group_1)) {
          continue;
        }

        for (auto def_group_0 : defs0) {
          auto def0 = def_group_0->front();
          auto def1 = def_group_1->front();
          if (exprsMap(def0, def1, false, mode)) {
            expr_prop.push_back(std::make_tuple(def0, def1, false));

            new_def_group.erase(def_group_0);
            new_def_group.erase(def_group_1);

            disjointExprsSet(mode).mapEntries(def0, def1);

            new_def_group.pushBack(
                disjointExprsSet(mode).disjointSetMap().at(def0));
          }
        }
      }
      unique_definitions_[mode][id_set] = new_def_group;
    } else {
      // Only one def has a nonzero entry
      unique_definitions_[mode][id_set] = defs0.size() > 0 ? defs0 : defs1;
    }
  }

  // Propagate on uses
  if (uses0.size() > 0 || uses1.size() > 0) {
    if (uses0.size() > 0 && uses1.size() > 0) {
      auto new_use_group = uses0;
      new_use_group.insert(uses1.begin(), uses1.end());

      for (auto use_group_1 : uses1) {
        if (uses0.has(use_group_1)) {
          continue;
        }

        for (auto use_group_0 : uses0) {
          auto use0 = use_group_0->front();
          auto use1 = use_group_1->front();
          if (exprsMap(use0, use1, true, mode)) {
            expr_prop.push_back(std::make_tuple(use0, use1, true));

            new_use_group.erase(use_group_0);
            new_use_group.erase(use_group_1);

            disjointExprsSet(mode).mapEntries(use0, use1);

            new_use_group.pushBack(
                disjointExprsSet(mode).disjointSetMap().at(use0));
          }
        }
      }
      unique_uses_[mode][id_set] = new_use_group;
    } else {
      // Only one use has a nonzero entry
      unique_uses_[mode][id_set] = uses0.size() > 0 ? uses0 : uses1;
    }
  }

  for (auto expr_tuple : expr_prop) {
    Expr* expr0;
    Expr* expr1;
    bool forward;
    std::tie(expr0, expr1, forward) = expr_tuple;
    mapThroughExpr(expr0, expr1, forward, mode);
  }
}

// Given first and second Exprs "match"
//   Expr type matches
//   IterDomain's in the inputs and outputs exact match, (including argument
//     position positions)
//   Paramters like Split's factor "match" (exact match on integers could be
//     better, as today it will just check it's the same symbol or evaluated to
//     the same constant. However, we know all the extents of all the
//     IterDomain's that exact map with eachother are the same value.
bool IterDomainGraph::mapThroughExpr(
    Expr* first,
    Expr* second,
    bool forward,
    IdMappingMode mode) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward, mode)) {
    return false;
  }

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->outputs() : first->inputs())
                       .vector();
  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->outputs() : second->inputs())
                        .vector();
  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : c10::irange(first_ids.size())) {
    mapIds(first_ids[out_i], second_ids[out_i], mode);
  }

  return true;
}

namespace {

// Returns the first pair of id's in ids detected to match eachother on the
// permissive map of the ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).view({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].view({2, 3})
// tv3 = tv1 + tv2
//
// Then we can see this overlap in the tv3 expression as:
//
// tv0 = { {0, 1, 2},
//         {3, 4, 5} }
//
// tv1 = { {0, 3},
//         {1, 4},
//         {2, 5} }
//
// tv2 = { {0, 1},
//         {2, 3},
//         {4, 5} }
//
// The elements in tv1 {3, 1, 4, 2}, map respectively to the elements in tv2 {1,
// 2, 3, 4}. The reason this is so important is it means that generating tv3 is
// no longer a trivially parallelizable problem (if we include the dag all the
// way to tv0). So tv0's axes cannot be inlined across both the tv0 and tv1
// path. This breaks some assumptions we have today in schedulers that will
// assume tv2 can be trivially inlined/parallelized. Instead we'd need to take
// into consideration the effective communication going on here, so that we pull
// multiple values of tv0 to compute tv3.
c10::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraph& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (id_graph.getDisjointIdsSet(mode).disjointSetMap().at(id1)->has(id2)) {
        return std::make_pair(id1, id2);
      }
    }
  }

  return {};
}

// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
findFirstSelfMapping(Fusion* fusion, const IterDomainGraph& id_graph) {
  for (auto tv : ir_utils::allTvs(fusion)) {
    // For each tensor, make sure root, rfactor and leaf domains
    // should not include domains that are mapped with another domain
    // in the same set of domains. This may be overly conservative,
    // and it maybe enough to check the root domains.

    // Root domains
    auto self_mappped_root_pair =
        detectMappablePair(tv->getRootDomain(), id_graph, IdMappingMode::EXACT);
    if (self_mappped_root_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_root_pair->first,
          self_mappped_root_pair->second,
          "Root");
    }

    // Rfactor domains
    if (tv->hasRFactor()) {
      auto self_mappped_rf_pair = detectMappablePair(
          tv->getRFactorDomain(), id_graph, IdMappingMode::EXACT);
      if (self_mappped_rf_pair.has_value()) {
        return std::make_tuple(
            tv,
            self_mappped_rf_pair->first,
            self_mappped_rf_pair->second,
            "RFactor");
      }
    }

    // Leaf domains
    auto self_mappped_leaf_pair = detectMappablePair(
        tv->domain()->domain(), id_graph, IdMappingMode::LOOP);
    if (self_mappped_leaf_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_leaf_pair->first,
          self_mappped_leaf_pair->second,
          "Leaf");
    }
  }
  return c10::nullopt;
}

} // namespace

// TODO: Should we avoid marking leaf Ids at this point?
void IterDomainGraph::initializeId(
    IterDomain* id,
    bool is_view_rfactor_id,
    bool is_leaf_id) {
  auto id_disjoint_set =
      disjointIdsSet(IdMappingMode::EXACT).initializeSet(id).first->second;

  if (id->definition() != nullptr) {
    auto expr_set = disjointExprsSet(IdMappingMode::EXACT)
                        .initializeSet(id->definition())
                        .first->second;
    unique_definitions_[IdMappingMode::EXACT][id_disjoint_set] = {expr_set};
  }

  auto use_it = id_uses_.find(id);
  if (use_it != id_uses_.end()) {
    auto use = use_it->second;
    if (use != nullptr) {
      auto expr_set = disjointExprsSet(IdMappingMode::EXACT)
                          .initializeSet(use)
                          .first->second;
      unique_uses_[IdMappingMode::EXACT][id_disjoint_set] = {expr_set};
    }
  }

  if (is_leaf_id) {
    disjointIdsSet(IdMappingMode::LOOP).initializeSet(id);
    if (id->definition() != nullptr) {
      disjointExprsSet(IdMappingMode::LOOP).initializeSet(id->definition());
    }
  }

  consumers_[id] = {};
  producers_[id] = {};

  if (is_view_rfactor_id) {
    view_rfactor_ids_.emplace(id);
  }
}

void IterDomainGraph::buildIterDomainUses(Fusion* fusion) {
  // Generate IterDomain uses:
  for (auto tv : ir_utils::allTvs(fusion)) {
    auto all_ids = ir_utils::allIDsOf(tv);
    for (auto id : all_ids) {
      if (id_uses_.find(id) == id_uses_.end()) {
        id_uses_[id] = nullptr;
      }

      auto def = id->definition();

      if (def == nullptr) {
        continue;
      }
      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        if (id_uses_.find(id) != id_uses_.end()) {
          TORCH_INTERNAL_ASSERT(
              id_uses_[id] == nullptr,
              "\nTried to set multiple uses to iteration domain: ",
              id->toString(),
              "\nWhich is not supported, tried to set expr:\n  ",
              def->toString(),
              "However the following expression was already set:\n  ",
              id_uses_[id]->toString());
        }
        id_uses_[inp_id] = def;
      }
    }
  }
}

void IterDomainGraph::initialIdProcessing(Fusion* fusion) {
  // Initialize entries for every iteration domain and mark view like
  // iteration domains and leaf iteration domains.
  for (auto tv : ir_utils::allTvs(fusion)) {
    const auto& domain = tv->domain()->domain();
    auto all_ids = ir_utils::allIDsOf(tv);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      bool is_view_rfactor_id = false;
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration
        // domain is marked as an rfactor product and is in the rfactor
        // domain, it's a view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->getMaybeRFactorDomain();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          is_view_rfactor_id = true;
        }
      }
      bool is_leaf_id =
          std::find(domain.begin(), domain.end(), id) != domain.end();
      initializeId(id, is_view_rfactor_id, is_leaf_id);
    }
  }
}

void IterDomainGraph::mapMultiOutput(Expr* expr) {
  auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
  if (std::distance(tv_outputs.begin(), tv_outputs.end()) <= 1) {
    // No multi TV outputs to map just return
    return;
  }

  TensorView* first_output_tv = *tv_outputs.begin();
  std::deque<TensorView*> other_tv_outputs(
      tv_outputs.begin(), tv_outputs.end());
  other_tv_outputs.pop_front();

  for (auto other_tv_output : other_tv_outputs) {
    // Map multi outputs of an expression to each other. c is current
    // output, and f as first output. Keep consistent with the later section
    // of producer and consumers. Which here producer is now "first output",
    // and consumer is still consumer. One exception is how the
    // domains left of CA positions are handled in the Parallel
    // map. Those domains are not mapped in producer and consumer
    // mappings as they do not share loops, but are mapped in the
    // case of mapping multiple outputs since they do share the
    // same loops.

    TORCH_INTERNAL_ASSERT(
        other_tv_output->getRootDomain().size() ==
            first_output_tv->getRootDomain().size(),
        "Multiple outputs with mismatched dimensions is not supported. ",
        "Only supported case is welford op where all outputs tvs have idential domains.");
    // other to first map
    std::unordered_map<IterDomain*, IterDomain*> o2f;
    for (const auto i : c10::irange(first_output_tv->getRootDomain().size())) {
      o2f.insert(std::make_pair(
          other_tv_output->getRootDomain()[i],
          first_output_tv->getRootDomain()[i]));
    }

    // Multi output mapping, outputs are required to have the same domain
    // and same transformations, so they can be mapped in permissive/exact,
    // and when within compute at position of domain()->domain() in the
    // parallel map.
    auto replay_FasC = BestEffortReplay(
        first_output_tv->domain()->domain(),
        other_tv_output->domain()->domain(),
        o2f);

    // Map the entire replay map between the multiple
    // consumers
    auto c2f_disjoint_sets = replay_FasC.getIterDomainEquivalence();
    for (auto disjoint_set : c2f_disjoint_sets.disjointSets()) {
      if (disjoint_set->empty()) {
        continue;
      }
      auto id0 = *disjoint_set->begin();
      for (auto id1 : disjoint_set->vector()) {
        mapIds(id0, id1, IdMappingMode::EXACT);
      }
    }

    // Map all entries for the Loop map as they share the same loops.
    for (auto f_id : first_output_tv->domain()->domain()) {
      auto disjoint_set = c2f_disjoint_sets.getDisjointSetOf(f_id);
      auto id0 = *(disjoint_set.begin());
      for (auto id1 : disjoint_set) {
        mapIds(id0, id1, IdMappingMode::LOOP);
      }
    }
  }
}

namespace {
//! Map corresponding inputs and outputs of swizzle op together
//!  on the given disjoint set, if the given id is an output
//!  of a swizzle operator.
//!
//! The current usage of swizzle operator is local to each tensor
//!  itself, so they should not affect exact or permissive mapping
//!  between iterdomains on different tensor domains.
//! TODO:
//!   Exact mapping based index hoisting of swizzled iterdomains
//!   is disabled currently and will be re-enabled in the next
//!   few build out steps.
void mapMaybeSwizzleOp(
    DisjointSets<IterDomain*>& disjoint_sets,
    IterDomain* id) {
  if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(id->definition())) {
    // Map each input to its corresponding output on the given
    // disjoint set if this is a loop swizzle. Loop swizzles don't impact
    // indexing, only iteration order.
    if (swizzle_2d->swizzleMode() == SwizzleMode::Loop) {
      disjoint_sets.mapEntries(swizzle_2d->inX(), swizzle_2d->outX());
      disjoint_sets.mapEntries(swizzle_2d->inY(), swizzle_2d->outY());
    }
  }
}
} // namespace

void IterDomainGraph::mapThroughLoopSwizzles(IdMappingMode mode) {
  for (auto use_it : id_uses_) {
    auto use = use_it.second;
    if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(use)) {
      // Map each input to its corresponding output on the given
      // disjoint set if this is a loop swizzle. Loop swizzles don't impact
      // indexing, only iteration order.
      if (swizzle_2d->swizzleMode() == SwizzleMode::Loop) {
        mapIds(swizzle_2d->inX(), swizzle_2d->outX(), mode);
        mapIds(swizzle_2d->inY(), swizzle_2d->outY(), mode);
      }
    }
  }
}

void IterDomainGraph::buildExactMap(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto p_tv : tv_inputs) {
      // For exact mapings do not map any broadcast dimensions to
      // non-broadcast dimensions. Prevent any broadcasted axes being mapped
      // to non-broadcasted axes.
      auto exact_c2p_root_map =
          PairwiseRootDomainMap(p_tv, c_tv, true)
              .mapConsumerToProducer(c_tv->domain(), p_tv->domain());

      for (auto c_id : getSortedKeys(exact_c2p_root_map, Statement::lessThan)) {
        auto p_id = exact_c2p_root_map.at(c_id);
        mapIds(c_id, p_id, IdMappingMode::EXACT);
      }

      // Same as permissive above but for exact
      auto exact_replay_PasC = BestEffortReplay(
          p_tv->domain()->domain(),
          c_tv->domain()->domain(),
          exact_c2p_root_map);

      const auto& exact_c2p_map = exact_replay_PasC.getReplay();

      for (auto c_id : getSortedKeys(exact_c2p_map, Statement::lessThan)) {
        auto p_id = exact_c2p_map.at(c_id);

        // TODO: consumers/producers should be on a per map basis, mapping
        // should include unique expr between the disjoint sets
        consumers_.at(p_id).pushBack(c_id);
        producers_.at(c_id).pushBack(p_id);
      }
    }

    mapThroughLoopSwizzles(IdMappingMode::EXACT);
  }
}

void IterDomainGraph::buildPermissiveMap(const std::vector<Expr*>& exprs) {
  copyGraph(IdMappingMode::EXACT, IdMappingMode::PERMISSIVE);

  for (auto expr : exprs) {
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      auto p_ids_vec = ir_utils::allIDsOf(p_tv);
      auto c_ids_vec = ir_utils::allIDsOf(c_tv);
      std::unordered_set<IterDomain*> p_ids(p_ids_vec.begin(), p_ids_vec.end());
      std::unordered_set<IterDomain*> c_ids(c_ids_vec.begin(), c_ids_vec.end());

      ForwardingInfo permissive_forwarding(p_tv, c_tv);
      for (auto entry : permissive_forwarding.producer_forwarding_map) {
        mapIds(entry.first, entry.second, IdMappingMode::PERMISSIVE);
      }

      for (auto entry : permissive_forwarding.consumer_forwarding_map) {
        mapIds(entry.first, entry.second, IdMappingMode::PERMISSIVE);
      }

      auto permissive_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv);

      for (auto entry : permissive_c2p_root_map.mapConsumerToProducer(
               c_tv->domain(), p_tv->domain())) {
        mapIds(entry.first, entry.second, IdMappingMode::PERMISSIVE);
      }
    }
  }
  mapThroughLoopSwizzles(IdMappingMode::PERMISSIVE);
}

void IterDomainGraph::mapRFactorExprs(Fusion* fusion) {
  // Explicitly map through rfactor transformations, if we have an op like:
  //
  // T1[x, y*z] = view(T0[x*y, z])
  // T3[x, y*z] = view(T2[x*y, z])
  // T4 = T0 + T2
  //
  // We want to map T1 and T3's rfactor transformations together by playing
  // the transformations forward since their root domains map. If instead we
  // have:
  //
  // T1[x, y*z] = view(T0[x*y, z])
  // T3[x, y*z] = view(T2[x*y, z])
  // T4 = T1 + T3
  //
  // Then we wouldn't have a mapping of T1 and T3's root domain, we'd have a
  // mapping of their rfactor domain, so we would want to map T1 and T3's
  // rfactor transformations starting at their rfactor domains.
  //
  // Therefore we'll explicitly map rfactor transformation iteration domains
  // forward and backwards. Something similar could happen with rfactor of
  // root domains, though it seems mapping rfactor reduction domains aren't
  // that important. Mapping view transformations is more important since view
  // is part of the compute definition so having the map through the
  // transformations makes it easy to check if different view operations are
  // consistent with eachother.

  auto all_tvs = ir_utils::allTvs(fusion);
  std::vector<TensorView*> all_consumer_tvs;
  std::copy_if(
      all_tvs.begin(),
      all_tvs.end(),
      std::back_inserter(all_consumer_tvs),
      [](TensorView* tv) { return !tv->isFusionInput() && tv->hasRFactor(); });

  // IterDomains could have multiple uses defined in the fusion if multiple
  // transformations were redefined (more than one transform propagation pass
  // was run and retransformed sections of the graph). We're going to make a
  // new uses map so we can easily process the actual uses of IterDomains. We
  // actually only need rfactor uses for this section of mapping, so we'll
  // limit this map to only rfactor transformations.
  std::unordered_map<IterDomain*, Expr*> rfactor_id_uses;

  // Order of traversal is important for processing all the rfactor ids as the
  // first pass will go forward through expressions and the second pass will
  // traverse backwards through them. ID's will be unique in this vector,
  // enforced when building it since it's built with rfactor_id_uses.
  std::vector<IterDomain*> rfactor_id_order;

  // Grab all the rfactor ids.
  for (auto consumer_tv : all_consumer_tvs) {
    auto exprs = StmtSort::getExprs(
        fusion,
        {consumer_tv->getMaybeRFactorDomain().begin(),
         consumer_tv->getMaybeRFactorDomain().end()});
    for (auto expr : exprs) {
      auto rfactor_inp_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
      TORCH_INTERNAL_ASSERT(
          expr->isA<Split>() || expr->isA<Merge>(),
          "Wasn't expecting the expression type of:\n",
          expr->toString(),
          "\nto be an expression defined in an rfactor transformation.");
      for (auto rfactor_inp_id : rfactor_inp_ids) {
        TORCH_INTERNAL_ASSERT(
            rfactor_id_uses.find(rfactor_inp_id) == rfactor_id_uses.end(),
            "Was expecting iter domains to only have one active transformation but found id ",
            rfactor_inp_id->toString(),
            " used in\n",
            rfactor_id_uses.at(rfactor_inp_id),
            "\nand\n",
            expr->toString());
        rfactor_id_uses.emplace(std::make_pair(rfactor_inp_id, expr));
        rfactor_id_order.push_back(rfactor_inp_id);
      }
    }
    for (auto rfactor_id : consumer_tv->getMaybeRFactorDomain()) {
      if (rfactor_id->isRFactorProduct()) {
        rfactor_id_uses.emplace(std::make_pair(rfactor_id, nullptr));
        rfactor_id_order.push_back(rfactor_id);
      }
    }
  }

  // if prop_forward we're going forward through transformations and
  // expressions, meaning if inputs of expressions map then we map their
  // outputs, otherwise we're traversing backwards, meaning if outputs of
  // expressions map then we map their inputs.
  for (auto prop_forward : {true, false}) {
    std::unordered_set<Expr*> visited_exprs;

    for (auto rfactor_id_i : c10::irange(rfactor_id_order.size())) {
      auto first_rfactor_id = prop_forward
          ? rfactor_id_order[rfactor_id_i]
          : rfactor_id_order[rfactor_id_order.size() - 1 - rfactor_id_i];

      // At should be safe since we made rfactor_id_order and rfactor_id_uses
      // at the same time so they should have the same exact entries.
      auto first_expr = prop_forward ? rfactor_id_uses.at(first_rfactor_id)
                                     : first_rfactor_id->definition();

      if (first_expr == nullptr) {
        continue;
      }

      if (visited_exprs.find(first_expr) != visited_exprs.end()) {
        continue;
      }
      visited_exprs.emplace(first_expr);

      // Only need to be concerned here with mapping across rfactor iter
      // domains, so isolate out those.
      auto all_exact_map_ids = disjointIdsSet(IdMappingMode::EXACT)
                                   .getDisjointSetOf(first_rfactor_id);
      std::vector<IterDomain*> exact_map_rf_ids;
      std::copy_if(
          all_exact_map_ids.vector().begin(),
          all_exact_map_ids.vector().end(),
          std::back_inserter(exact_map_rf_ids),
          [](IterDomain* id) { return id->isRFactorProduct(); });

      for (auto exact_map_rf_id : exact_map_rf_ids) {
        if (exact_map_rf_id == first_rfactor_id) {
          continue;
        }
        // If there's an input with an rfactor domain we could have an exact
        // mapped rfactor id that's on the input meaning it wouldn't have an
        // entry in rfactor_id_uses
        auto other_use =
            rfactor_id_uses.find(exact_map_rf_id) == rfactor_id_uses.end()
            ? nullptr
            : rfactor_id_uses.at(exact_map_rf_id);
        auto other_expr =
            prop_forward ? other_use : exact_map_rf_id->definition();

        if (other_expr == nullptr) {
          continue;
        }

        if (visited_exprs.find(other_expr) != visited_exprs.end()) {
          continue;
        }

        mapThroughExpr(
            first_expr, other_expr, prop_forward, IdMappingMode::EXACT);
      }
    }
  }
}

void IterDomainGraph::buildAlmostExactMap() {
  // Build almost exact map by forwarding through broadcast axes
  copyGraph(IdMappingMode::EXACT, IdMappingMode::ALMOSTEXACT);

  std::unordered_set<Expr*> visited;
  auto all_elements = disjointIdsSet(IdMappingMode::EXACT).getAllElements();
  for (auto entry : all_elements.vector()) {
    if (entry->definition() == nullptr) {
      continue;
    }
    auto def = entry->definition();
    if (!visited.emplace(def).second) {
      continue;
    }
    if (auto merge = dynamic_cast<Merge*>(def)) {
      if (merge->inner()->extent()->isOneInt()) {
        mapIds(merge->outer(), merge->out(), IdMappingMode::ALMOSTEXACT);
      }
      if (merge->outer()->extent()->isOneInt()) {
        mapIds(merge->inner(), merge->out(), IdMappingMode::ALMOSTEXACT);
      }
    } else if (auto split = dynamic_cast<Split*>(def)) {
      if (split->factor()->isOneInt() && split->startOffset()->isZeroInt() &&
          split->stopOffset()->isZeroInt()) {
        if (split->innerSplit()) {
          mapIds(split->in(), split->outer(), IdMappingMode::ALMOSTEXACT);
        } else {
          mapIds(split->in(), split->inner(), IdMappingMode::ALMOSTEXACT);
        }
      }
    }
  }
}

void IterDomainGraph::buildLoopMap(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      auto p_ids_vec = ir_utils::allIDsOf(p_tv);
      auto c_ids_vec = ir_utils::allIDsOf(c_tv);
      std::unordered_set<IterDomain*> p_ids(p_ids_vec.begin(), p_ids_vec.end());
      std::unordered_set<IterDomain*> c_ids(c_ids_vec.begin(), c_ids_vec.end());

      auto permissive_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv);

      // Look for matching ID transformations in producer and consumer, replay
      // producer as consumer. We use the symmetric API of BestEffortReplay so
      // that both broadcast and squeeze are handled correctly.
      const auto permissive_disjoint_sets =
          BestEffortReplay::replayPasC(p_tv, c_tv, -1, permissive_c2p_root_map)
              .getIterDomainEquivalence();

      for (auto& dset : permissive_disjoint_sets.disjointSets()) {
        auto& vec = dset->vector();
        for (auto i : c10::irange(vec.size())) {
          auto id1 = vec[i];
          for (auto j : c10::irange(i + 1, vec.size())) {
            auto id2 = vec[j];
            if (p_ids.count(id1) && c_ids.count(id2)) {
              consumers_.at(id1).pushBack(id2);
              producers_.at(id2).pushBack(id1);
              if (idIsAComputeAtLeafDomain(id1, p_tv, c_tv) &&
                  idIsALeafDomain(id2, c_tv)) {
                mapIds(id1, id2, IdMappingMode::LOOP);
              }
            }
            if (c_ids.count(id1) && p_ids.count(id2)) {
              producers_.at(id1).pushBack(id2);
              consumers_.at(id2).pushBack(id1);
              if (idIsAComputeAtLeafDomain(id2, p_tv, c_tv) &&
                  idIsALeafDomain(id1, c_tv)) {
                mapIds(id1, id2, IdMappingMode::LOOP);
              }
            }
          }
        }
      }
    }
  }
}

void IterDomainGraph::build(Fusion* fusion) {
  FusionGuard fg(fusion);

  // Add uses to all iter domains.
  buildIterDomainUses(fusion);

  // Initialize the maps with all the IterDomains defined in the fusion.
  initialIdProcessing(fusion);

  // Filter non-TensorView expressions
  auto all_exprs = fusion->exprs();
  std::vector<Expr*> tv_exprs;

  std::copy_if(
      all_exprs.begin(),
      all_exprs.end(),
      std::back_inserter(tv_exprs),
      [](Expr* expr) { return ir_utils::isTvOp(expr); });

  for (auto expr : tv_exprs) {
    // Connect multi-output expressions as they're trivial to connect.
    mapMultiOutput(expr);
  }

  buildExactMap(tv_exprs);
  // Map forward and backward through TV root<->rfactor to cross map
  // connections that are not explicitly defined through input<->output
  // expression maps.
  //
  // Updates both permissive and exact mapping, must be done after exact and
  // permissive maps are built but before we copy the exact map for the almost
  // exact map.
  mapRFactorExprs(fusion);

  buildAlmostExactMap();
  buildPermissiveMap(tv_exprs);
  buildAlmostExactMap();
  buildLoopMap(tv_exprs);

  // Debug, make sure there's no self mapping in TensorView's during lowering
  // that would invalidate lowering assumptions.
  self_mapping_info_ = findFirstSelfMapping(fusion, *this);
}

void IterDomainGraph::copyGraph(
    IdMappingMode from_mode,
    IdMappingMode to_mode) {
  if (from_mode == to_mode) {
    return;
  }

  disjointIdsSet(to_mode) = disjointIdsSet(from_mode);
  disjointExprsSet(to_mode) = disjointExprsSet(from_mode);
  unique_definitions_[to_mode] = {};
  unique_uses_[to_mode] = {};

  for (auto is_defs : std::vector<bool>({true, false})) {
    if (is_defs) {
      if (unique_definitions_.find(from_mode) == unique_definitions_.end()) {
        continue;
      }
    } else {
      if (unique_uses_.find(from_mode) == unique_uses_.end()) {
        continue;
      }
    }
    auto& from_defs_or_uses =
        is_defs ? unique_definitions_[from_mode] : unique_uses_[from_mode];

    auto& to_defs_or_uses =
        is_defs ? unique_definitions_[to_mode] : unique_uses_[to_mode];

    for (auto entry : from_defs_or_uses) {
      // Mappings from IterDomain to a vector of disjoint expression sets
      auto orig_id = entry.first->front();
      auto orig_expr_sets = entry.second;

      auto new_id_set = disjointIdsSet(to_mode).disjointSetMap().at(orig_id);

      VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<Expr*>>>
          new_exprs;

      for (auto orig_expr_set : orig_expr_sets.vector()) {
        auto orig_expr = orig_expr_set->front();
        auto new_expr_set =
            disjointExprsSet(to_mode).disjointSetMap().at(orig_expr);
        new_exprs.pushBack(new_expr_set);
      }

      if (new_exprs.size() > 0) {
        to_defs_or_uses[new_id_set] = new_exprs;
      }
    }
  }
}

ComputeAtMap::ComputeAtMap(Fusion* fusion)
    : id_graph_(fusion), concretized_bcasts_(fusion), fusion_(fusion) {
  build(fusion);
}

void ComputeAtMap::build(Fusion* fusion) {
  buildUniqueExactExprMaps();
  buildConcreteIds();
}

void ComputeAtMap::validateAndPropagatePType() {
  for (const auto& loop_disjoint_set :
       id_graph_.getDisjointIdsSet(IdMappingMode::LOOP).disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (auto id : loop_disjoint_set->vector()) {
      auto id_ptype = id->getParallelType();
      TORCH_INTERNAL_ASSERT(
          id_ptype == common_ptype || id_ptype == ParallelType::Serial ||
              common_ptype == ParallelType::Serial,
          "Issue validating parallel type disjoint ptype is, ",
          common_ptype,
          " but found in the set the id: ",
          id->toString());
      common_ptype =
          common_ptype == ParallelType::Serial ? id_ptype : common_ptype;
    }

    for (auto id : loop_disjoint_set->vector()) {
      id->parallelize(common_ptype);
    }
  }
}

void ComputeAtMap::allocateIndexVariables() {
  // Run through all disjoint sets registered in loop map,
  //  all lowered kir::ForLoop will correspond to one of the disjoint sets
  //  and we only need one index variable for each set.
  for (const auto& loop_disjoint_set :
       id_graph_.getDisjointIdsSet(IdMappingMode::LOOP).disjointSets()) {
    ParallelType ptype;
    // first allocate thread and grid parallel indices:
    //  The validation pass will check that the parallel bindings within the
    //  loop disjoint IDs set are consistent so all the loops within this
    //  disjoint set will be realized implicitly using parallel index
    //  variables.
    if (std::any_of(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [&ptype](IterDomain* id) {
              if (id->isThread() &&
                  // Halo extended parallel loops currently are handled
                  // differently and an index variable would still
                  // be allocated in this case.
                  (GpuLower::current()->haloInfo()->getExtent(id) == nullptr)) {
                ptype = id->getParallelType();
                return true;
              }
              return false;
            })) {
      loop_index_variable_map_[loop_disjoint_set.get()] =
          NamedScalar::getParallelIndex(ptype);
      continue;
    }

    // All loops in this set are non-parallel, non-concretized broadcast
    //  iterdomains, their "index variable" should be zero.
    if (std::all_of(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [](IterDomain* id) { return id->isBroadcast(); })) {
      loop_index_variable_map_[loop_disjoint_set.get()] = fusion_->zeroVal();
      continue;
    }

    // Allocate variable for the iterdomains:
    auto concrete_loop_id_it = concrete_id_cache_.find(loop_disjoint_set);
    TORCH_INTERNAL_ASSERT(
        concrete_loop_id_it != concrete_id_cache_.end(),
        "Concrete id not computed");

    auto concrete_loop_id = concrete_loop_id_it->second;

    // Need to allocate double buffered loop differently.
    if (GpuLower::current()->doubleBufferInfo().isDoubleBufferedIterDomain(
            concrete_loop_id)) {
      // Allocate index variable for each stage of the double buffered loop.
      double_buffered_loop_index_variable_map_[loop_disjoint_set.get()] =
          std::make_unique<DoubleBufferIndices>(DoubleBufferIndices(
              {{DoubleBufferLoopStage::Prolog,
                IrBuilder::create<Int>(c10::nullopt)},
               {DoubleBufferLoopStage::Main,
                IrBuilder::create<Int>(c10::nullopt)},
               {DoubleBufferLoopStage::Epilog,
                IrBuilder::create<Int>(c10::nullopt)}}));
    } else {
      // Everything now should be serial concrete loops,
      //   we just allocate a loop index integer for each set of loops.
      loop_index_variable_map_[loop_disjoint_set.get()] =
          IrBuilder::create<Int>(c10::nullopt);
    }
  }
}

Val* ComputeAtMap::getIndexVariable(
    IterDomain* id,
    DoubleBufferLoopStage double_buffer_loop_stage) const {
  TORCH_INTERNAL_ASSERT(
      id_graph_.getDisjointIdsSet(IdMappingMode::LOOP).mappingExists(id),
      "Index Variable: no index variable allocated as ",
      id->toString(),
      " is not registered in loop map");
  const auto* loop_set =
      &(id_graph_.getDisjointIdsSet(IdMappingMode::LOOP).getDisjointSetOf(id));

  // Check if this loop was modified by double buffer pass.
  bool is_double_buffer_iterdomain =
      GpuLower::current()->doubleBufferInfo().isDoubleBufferedIterDomain(id);

  if (is_double_buffer_iterdomain) {
    // Use dedicated double buffer index variable if the loop is double buffer
    // loop
    if (double_buffer_loop_stage == DoubleBufferLoopStage::NotApplicable) {
      // The double buffered loop stages are created after the loop nest
      //  lowering phase so this function will be querried before the double
      //  buffer pass. At that point, no forloop has any double buffer
      //  stage defined, and we just default to using the main stage index.
      double_buffer_loop_stage = DoubleBufferLoopStage::Main;
    }
    return double_buffered_loop_index_variable_map_.at(loop_set)->at(
        double_buffer_loop_stage);
  } else {
    return loop_index_variable_map_.at(loop_set);
  }
}

IterDomain* ComputeAtMap::computeConcreteId(
    IterDomain* id,
    IdMappingMode mode) {
  const auto& disjoint_set_shared_ptr = disjointSetOf(id, mode);

  TORCH_INTERNAL_ASSERT(
      disjoint_set_shared_ptr->vector().size(),
      "Empty disjoint set found for ",
      id->toString());

  if (disjoint_set_shared_ptr->vector().size() == 1) {
    // If only one entry in the disjoint set, by definition the existing ID
    // has to be the concrete ID.
    return disjoint_set_shared_ptr->vector().front();
  }

  // Grab a set of candidate concrete_ids, we track towards the consumers in
  // the ID group as one of those is guaranteed to be a valid concrete id.
  VectorOfUniqueEntries<IterDomain*> maybe_concrete_ids;
  for (auto id : disjoint_set_shared_ptr->vector()) {
    bool id_output = true;
    for (auto consumer_id : id_graph_.consumers().at(id).vector()) {
      if (disjoint_set_shared_ptr->has(consumer_id)) {
        id_output = false;
        break;
      }
    }
    if (id_output) {
      maybe_concrete_ids.pushBack(id);
    }
  }

  // Shouldn't ever happen, it would mean there's an error somewhere in the
  // graph.
  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.vector().size() > 0,
      "No potential concrete_id's found for ",
      id->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  // Broadcast resolution is what we have to figure out here. So if we
  // traverse back from leaves to rfactor inputs through the exact map, if
  // there's an operation with a broadcast input that's resolved within the
  // history all of the domains in all of the maybe_rfactor_ids, then the
  // concrete ID must resolve that broadcast.
  //
  // (1) Compute "traversed IDs" which is every exact disjoint set starting at
  // all maybe concrete ID's traversing back through exact map.
  //
  // (2) Check all broadcast sets, remove from "traversed IDs" any broadcast
  // set that has its broadcast resolved ID within "traversed IDs", and all
  // IterDomains dependant on that broadcast.
  //
  // (3) Start at all "traversed IDs" set that has an rfactor domain, traverse
  // backwards to inputs and remove every exact ID set from "traversed IDs".
  //
  // Remove (2) and (3) from (1) and we have the iteration domains we must
  // resolve. The concrete ID must be in that set.
  //
  // Find any maybe concrete ID through the same iter/broadcast counting as
  // before as it should work fine.

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      maybe_concrete_exact_sets;

  for (auto maybe_concrete_id : maybe_concrete_ids) {
    maybe_concrete_exact_sets.pushBack(
        disjointSetOf(maybe_concrete_id, IdMappingMode::EXACT));
  }

  // Going to iteratively modify this to be all sets that the concrete ID
  // needs to cover
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      all_exact_sets_covered =
          getAllDisjointSetProducers(maybe_concrete_exact_sets);

  // Remove all broadcast domains that are resolved within the history of any
  // of the maybe concrete sets.
  {
    // All broadcast exact sets in all_exact_sets_covered that are resolved by
    // IterDomains in all_exact_sets_covered
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        resolved_broadcasts;

    for (auto exact_set : all_exact_sets_covered) {
      TORCH_INTERNAL_ASSERT(
          exact_set->vector().size(),
          "Cannot compute concrete id of empty set.");
      auto c_id = getConcreteMappedID(
          exact_set->vector().front(), IdMappingMode::EXACT);

      if (!c_id->isBroadcast()) {
        continue;
      }

      bool concretized_in_group = false;
      for (auto bcast_id : exact_set->vector()) {
        auto concretized_ids =
            concretized_bcasts_.allConcretizedDomains(bcast_id);
        for (auto concretized_id : concretized_ids) {
          if (all_exact_sets_covered.has(
                  disjointSetOf(concretized_id, IdMappingMode::EXACT))) {
            concretized_in_group = true;
            break;
          }
        }
        if (concretized_in_group) {
          break;
        }
      }

      if (concretized_in_group) {
        resolved_broadcasts.pushBack(exact_set);
      }
    }

    // Need to remove all uses of broadcast dims that are resolved in this
    // group, and all their uses.
    auto all_resolved_broadcast_uses =
        getAllDisjointSetConsumers(resolved_broadcasts);

    // Remove broadcast resolved sets from all_exact_sets_covered by
    // effectively doing an inplace copy_if
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        tmp_all_exact_sets_covered;
    std::swap(tmp_all_exact_sets_covered, all_exact_sets_covered);
    for (auto entry : tmp_all_exact_sets_covered) {
      if (all_resolved_broadcast_uses.has(entry)) {
        continue;
      }
      all_exact_sets_covered.pushBack(entry);
    }
  }

  // Remove all domains in the history of sets marked as rfactor.
  {
    // All exact sets in the history of an rfactored domain
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        produces_rfactor_dom;
    for (auto exact_set : all_exact_sets_covered) {
      if (produces_rfactor_dom.has(exact_set)) {
        // Already processed
        continue;
      }
      if (std::none_of(
              exact_set->vector().begin(),
              exact_set->vector().end(),
              [&](IterDomain* id) { return isViewRfactor(id); })) {
        continue;
      }
      VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
          rfactor_history = getAllDisjointSetProducers({exact_set});
      for (auto entry : rfactor_history) {
        // Leave rfactor exact set, unless it's in the history of another
        // rfactor domain.
        if (entry != exact_set) {
          produces_rfactor_dom.pushBack(entry);
        }
      }
    }

    // Remove all sets in rfactor history from all_exact_sets_covered by
    // effectively doing an inplace copy_if
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        tmp_all_exact_sets_covered;
    std::swap(tmp_all_exact_sets_covered, all_exact_sets_covered);
    for (auto entry : tmp_all_exact_sets_covered) {
      if (produces_rfactor_dom.has(entry)) {
        continue;
      }
      all_exact_sets_covered.pushBack(entry);
    }
  }

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      input_ids;

  {
    // Remove any concrete id that's not still in all_exact_sets_covered,
    // basically copy_if
    decltype(maybe_concrete_ids) tmp_maybe_concrete_ids;
    std::swap(maybe_concrete_ids, tmp_maybe_concrete_ids);
    for (auto entry : tmp_maybe_concrete_ids) {
      if (all_exact_sets_covered.has(
              disjointSetOf(entry, IdMappingMode::EXACT))) {
        maybe_concrete_ids.pushBack(entry);
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.vector().size() > 0,
      "No potential concrete_id's found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  // The concrete_id should have the most roots it can trace back to that are
  // iter domains, (non-broadcast/non-reduction). We don't trace back through
  // view operations, so the one with the most iter root domains is the
  // concrete ID.
  IterDomain* concrete_id = nullptr;
  int max_iter_root_count = 0;
  int max_bcast_root_count = 0;

  for (auto maybe_concrete_id : maybe_concrete_ids.vector()) {
    auto concrete_id_root_sets = getInputDisjointSetsOf(maybe_concrete_id);

    int bcast_root_count = std::count_if(
        concrete_id_root_sets.vector().begin(),
        concrete_id_root_sets.vector().end(),
        [&](std::shared_ptr<VectorOfUniqueEntries<IterDomain*>> set) {
          return set->vector()[0]->isBroadcast();
        });

    int iter_root_count =
        (int)concrete_id_root_sets.vector().size() - bcast_root_count;
    if (iter_root_count > max_iter_root_count ||
        (iter_root_count == max_iter_root_count &&
         bcast_root_count > max_bcast_root_count)) {
      max_iter_root_count = iter_root_count;
      max_bcast_root_count = bcast_root_count;
      concrete_id = maybe_concrete_id;
    }
  }

  TORCH_INTERNAL_ASSERT(
      concrete_id != nullptr,
      "No concrete_id found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  return concrete_id;
}

void ComputeAtMap::buildConcreteIds() {
  // For the exact map just select the first ID since they're all exactly the
  // same size, it doesn't matter which is selected. This should be run-to-run
  // deterministic but which ID gets selected her depends on the traversal
  // order generating the set (compute at map build).
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::EXACT).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    concrete_id_cache_[disjoint_set_shared_ptr] = first_id;
  }

  // The following two algorithms seem quite wasteful. Should find a more
  // efficient way to compute concrete IDs.
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::PERMISSIVE).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  // Same as exact computation
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::ALMOSTEXACT).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::ALMOSTEXACT);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::LOOP).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }
}

bool ComputeAtMap::areExactExprs(Expr* expr_1, Expr* expr_2) {
  if (typeid(*expr_1) != typeid(*expr_2)) {
    return false;
  }

  if (expr_1->isA<Swizzle2D>()) {
    auto swizzle_1 = expr_1->as<Swizzle2D>();
    auto swizzle_2 = expr_2->as<Swizzle2D>();
    if (swizzle_1->swizzleType() != swizzle_2->swizzleType() ||
        swizzle_1->swizzleMode() != swizzle_2->swizzleMode()) {
      return false;
    }
  }

  TORCH_INTERNAL_ASSERT(
      expr_1->inputs().size() == expr_2->inputs().size() &&
          expr_1->outputs().size() == expr_2->outputs().size(),
      "Expr traversal doesn't support variable number of inputs and outputs.");

  for (auto input_i : c10::irange(expr_1->inputs().size())) {
    if (expr_1->inputs()[input_i]->isA<IterDomain>() &&
        !areMapped(
            expr_1->inputs()[input_i]->as<IterDomain>(),
            expr_2->inputs()[input_i]->as<IterDomain>(),
            IdMappingMode::EXACT)) {
      // Inputs don't exact map in the right order
      return false;
    }
  }

  for (auto output_i : c10::irange(expr_1->outputs().size())) {
    if (expr_1->outputs()[output_i]->isA<IterDomain>() &&
        !areMapped(
            expr_1->outputs()[output_i]->as<IterDomain>(),
            expr_2->outputs()[output_i]->as<IterDomain>(),
            IdMappingMode::EXACT)) {
      // Outputs don't exact map in the right order
      return false;
    }
  }
  // Expr's have exact mapped inputs and outputs, including parameters of the
  // transformation.
  return true;
}

void ComputeAtMap::buildUniqueExactExprMaps() {
  // Start by building definitions
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::EXACT).disjointSets()) {
    std::vector<Expr*> definitions;

    // N^2 in number of unique transformations, this might be better to do
    // when generating the map.
    for (auto id : disjoint_set_shared_ptr->vector()) {
      if (id->definition() != nullptr) {
        auto id_inputs =
            ir_utils::filterByType<IterDomain>(id->definition()->inputs());
        if (std::any_of(id_inputs.begin(), id_inputs.end(), [&](auto id_input) {
              return disjoint_set_shared_ptr->has(id_input);
            })) {
          // Definition to this exact map, shouldn't be marked as a definition
          // to traverse on the exact map.

          // This is a WAR for FusionSimpleSwizzle2_CUDA wher there is a
          // pattern like:
          //
          // tv0[32, 32]
          // tv0->swizzle(Swizzle2DType::ZShape, 0, 1);
          //
          // each root domain is exact mapped with the outputs of the swizzle.
          // So the pre and post swizzle ID is in an exact set, but that exact
          // set also has the swizzle as a definition that leads to itself.
          //
          // TODO: Try to formalize this better in the exact ID traversal.
          // Right now its just interfering with concrete ID detection.
          continue;
        }
        bool match = false;
        for (auto recorded_def : definitions) {
          if (areExactExprs(id->definition(), recorded_def)) {
            match = true;
            break;
          }
        }
        if (!match) {
          definitions.push_back(id->definition());
        }
      }
    }
    unique_exact_definitions_[disjoint_set_shared_ptr] = definitions;
  }

  // Use definitions to build uses
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::EXACT).disjointSets()) {
    // Make sure uses is always initialized even there are no uses.
    if (unique_exact_uses_.find(disjoint_set_shared_ptr) ==
        unique_exact_uses_.end()) {
      unique_exact_uses_[disjoint_set_shared_ptr] = {};
    }

    auto definition_it =
        unique_exact_definitions_.find(disjoint_set_shared_ptr);

    if (definition_it == unique_exact_definitions_.end()) {
      continue;
    }

    const auto& definitions = definition_it->second;

    for (auto definition : definitions) {
      auto inp_ids = ir_utils::filterByType<IterDomain>(definition->inputs());
      for (auto inp : inp_ids) {
        auto inp_disjoint_set_shared_ptr =
            disjointSetOf(inp, IdMappingMode::EXACT);
        // Initialize uses entry
        if (unique_exact_uses_.find(inp_disjoint_set_shared_ptr) ==
            unique_exact_uses_.end()) {
          unique_exact_uses_[inp_disjoint_set_shared_ptr] = {};
        }

        auto& uses = unique_exact_uses_.at(inp_disjoint_set_shared_ptr);

        bool already_added = false;
        for (auto other_use : uses) {
          if (areExactExprs(definition, other_use)) {
            already_added = true;
            break;
          }
        }
        if (already_added) {
          continue;
        }

        if (!already_added) {
          uses.push_back(definition);
        }
      }
    }
  }
}

IterDomain* ComputeAtMap::getConcreteMappedID(
    IterDomain* id,
    IdMappingMode mode) const {
  auto disjoint_set_shared_ptr = disjointSetOf(id, mode);

  TORCH_INTERNAL_ASSERT(
      disjoint_set_shared_ptr->vector().size() > 0,
      "Empty disjoint set found for ",
      id->toString());

  auto cache_it = concrete_id_cache_.find(disjoint_set_shared_ptr);

  TORCH_INTERNAL_ASSERT(
      cache_it != concrete_id_cache_.end(),
      "Could not find concrete id for: ",
      id->toString(),
      " with mode ",
      mode);

  return cache_it->second;
}

namespace {

std::string idGraphDisjointIdSetToString(
    const ComputeAtMap& ca_map,
    IdMappingMode mode) {
  std::stringstream ss;
  // Sort vectors before printing so that the resulting output is
  // printed deterministically
  auto disjoint_sets = ca_map.getIdSets(mode).disjointSets();
  std::sort(
      disjoint_sets.begin(),
      disjoint_sets.end(),
      [&](const auto& set1, const auto& set2) {
        if (set1->empty()) {
          return true;
        } else if (set2->empty()) {
          return false;
        } else {
          auto concrete_id1 = ca_map.getConcreteMappedID(set1->front(), mode);
          auto concrete_id2 = ca_map.getConcreteMappedID(set2->front(), mode);
          return Statement::lessThan(concrete_id1, concrete_id2);
        }
      });
  for (const auto& s_ptr : disjoint_sets) {
    const auto& set = *s_ptr;
    IterDomain* concrete_id = nullptr;
    if (!set.empty()) {
      auto id = set.front();
      concrete_id = ca_map.getConcreteMappedID(id, mode);
    }
    ss << "  {";
    for (auto entry : set.vector()) {
      ss << abstractToString(entry);
      if (entry == concrete_id) {
        ss << "*";
      }
      if (entry != set.back()) {
        ss << "; ";
      }
    }
    ss << " }\n";
  }
  return ss.str();
}

} // namespace

std::string ComputeAtMap::toString() const {
  std::stringstream ss;
  ss << "Compute at map { \n";
  ss << "Exact map:\n"
     << idGraphDisjointIdSetToString(*this, IdMappingMode::EXACT);
  ss << "Almost Exact map:\n"
     << idGraphDisjointIdSetToString(*this, IdMappingMode::ALMOSTEXACT);
  ss << "Loop map:\n"
     << idGraphDisjointIdSetToString(*this, IdMappingMode::LOOP);
  ss << "Permissive map:\n"
     << idGraphDisjointIdSetToString(*this, IdMappingMode::PERMISSIVE);
  ss << "Consumer maps:\n";
  for (auto key : getSortedKeys(id_graph_.consumers(), Statement::lessThan)) {
    auto consumers = id_graph_.consumers().at(key);
    std::sort(consumers.begin(), consumers.end(), Statement::lessThan);
    ss << "  " << key->toString() << " :: " << consumers.toString() << "\n";
  }

  ss << "Producer maps:\n";
  for (auto key : getSortedKeys(id_graph_.producers(), Statement::lessThan)) {
    VectorOfUniqueEntries<IterDomain*> producers =
        id_graph_.producers().at(key);
    std::sort(producers.begin(), producers.end(), Statement::lessThan);
    ss << "  " << key->toString() << " :: " << producers.toString() << "\n";
  }

  ss << "} compute at map" << std::endl;
  return ss.str();
}

bool ComputeAtMap::isViewRfactor(IterDomain* ref_id) const {
  return id_graph_.viewRfactorIds().find(ref_id) !=
      id_graph_.viewRfactorIds().end();
}

std::vector<IterDomain*> ComputeAtMap::getViewRfactorDomainsOfIdGroup(
    IterDomain* ref_id,
    IdMappingMode mode) const {
  auto disjoint_set = disjointSetOf(ref_id, mode);
  std::vector<IterDomain*> rfactor_ids;
  for (auto disjoint_id : disjoint_set->vector()) {
    if (id_graph_.viewRfactorIds().find(disjoint_id) !=
        id_graph_.viewRfactorIds().end()) {
      rfactor_ids.push_back(disjoint_id);
    }
  }
  return rfactor_ids;
}

const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>& ComputeAtMap::
    disjointSetOf(IterDomain* id, IdMappingMode mode) const {
  TORCH_INTERNAL_ASSERT(
      idExistsInMap(id),
      id->toString(),
      " has not been processed in this Compute At Map, yet the disjoint set for it was requested.");
  return getIdSets(mode).disjointSetMap().at(id);
}

const DisjointSets<IterDomain*>& ComputeAtMap::getIdSets(
    IdMappingMode mode) const {
  return id_graph_.getDisjointIdsSet(mode);
}

bool ComputeAtMap::idExistsInMap(IterDomain* id) const {
  return getIdSets(IdMappingMode::EXACT).disjointSetMap().find(id) !=
      getIdSets(IdMappingMode::EXACT).disjointSetMap().end();
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getInputDisjointSetsOf(IterDomain* of_id, bool stop_at_rfactor) {
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      input_disjoint_sets;

  VectorOfUniqueEntries<IterDomain*> inputs;
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {disjointSetOf(of_id, IdMappingMode::EXACT)});
  std::unordered_set<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;
  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.emplace(currently_visiting).second) {
      continue;
    }
    auto defs_it = unique_exact_definitions_.find(currently_visiting);
    TORCH_INTERNAL_ASSERT(
        defs_it != unique_exact_definitions_.end(),
        "unique_exact_definitions_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // If there's no definition, we've found an input.
    if (defs_it->second.empty()) {
      input_disjoint_sets.pushBack(currently_visiting);
      continue;
    }

    if (stop_at_rfactor &&
        std::any_of(
            currently_visiting->vector().begin(),
            currently_visiting->vector().end(),
            [&](IterDomain* id) { return isViewRfactor(id); })) {
      input_disjoint_sets.pushBack(currently_visiting);
      continue;
    }

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        producers_of_currently_visiting;

    for (auto def : defs_it->second) {
      auto id_inps = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto id_inp : id_inps) {
        producers_of_currently_visiting.pushBack(
            disjointSetOf(id_inp, IdMappingMode::EXACT));
      }
    }

    // Add producers to visit if not already there
    for (auto producer : producers_of_currently_visiting.vector()) {
      if (visited.find(producer) == visited.end()) {
        to_visit.push_back(producer);
      }
    }
  }

  return input_disjoint_sets;
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getAllDisjointSetProducers(
    const VectorOfUniqueEntries<
        std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets) {
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto defs_it = unique_exact_definitions_.find(currently_visiting);
    TORCH_INTERNAL_ASSERT(
        defs_it != unique_exact_definitions_.end(),
        "unique_exact_definitions_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        producers_of_currently_visiting;

    for (auto def : defs_it->second) {
      auto id_inps = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto id_inp : id_inps) {
        producers_of_currently_visiting.pushBack(
            disjointSetOf(id_inp, IdMappingMode::EXACT));
      }
    }

    // Add producers to visit if not already there
    for (auto producer : producers_of_currently_visiting.vector()) {
      if (!visited.has(producer)) {
        to_visit.push_back(producer);
      }
    }
  }

  return visited;
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getAllDisjointSetConsumers(
    const VectorOfUniqueEntries<
        std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets) {
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto uses_it = unique_exact_uses_.find(currently_visiting);
    TORCH_INTERNAL_ASSERT(
        uses_it != unique_exact_uses_.end(),
        "unique_exact_uses_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // Traverse consumers of current disjoint set and collect unique exact
    // disjoint set consumers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        consumers_of_currently_visiting;

    for (auto uses : uses_it->second) {
      auto id_outs = ir_utils::filterByType<IterDomain>(uses->outputs());
      for (auto id_out : id_outs) {
        consumers_of_currently_visiting.pushBack(
            disjointSetOf(id_out, IdMappingMode::EXACT));
      }
    }

    // Add consumers to visit if not already there
    for (auto consumer : consumers_of_currently_visiting.vector()) {
      if (!visited.has(consumer)) {
        to_visit.push_back(consumer);
      }
    }
  }

  return visited;
}

void IterDomainGraph::updateComputeWith(TensorView* compute_with_tv) {
  TORCH_INTERNAL_ASSERT(
      compute_with_tv->hasResolvedComputeWith(),
      "Invalid tensor: ",
      compute_with_tv->toString());

  // Can use any consumer this tensor is computed with
  auto consumer_tv = compute_with_tv->getComputeWithConsumers().at(0);

  for (auto pos = compute_with_tv->getComputeAtPosition();
       pos < compute_with_tv->getComputeWithPosition();
       ++pos) {
    auto id = compute_with_tv->axis(pos);

    // Find the matching consumer ID using the permissive map
    auto it = std::find_if(
        consumer_tv->domain()->domain().begin(),
        consumer_tv->domain()->domain().end(),
        [&](auto consumer_id) {
          return getDisjointIdsSet(IdMappingMode::PERMISSIVE)
              .disjointSetMap()
              .at(id)
              ->has(consumer_id);
        });
    TORCH_INTERNAL_ASSERT(
        it != consumer_tv->domain()->domain().end(),
        "No consumer leaf ID of tensor ",
        consumer_tv->toString(),
        " permissively mapped with: ",
        id->toString());

    IterDomain* consumer_id = *it;

    mapIds(id, consumer_id, IdMappingMode::LOOP);
  }
}

void ComputeAtMap::updateComputeWith(TensorView* compute_with_tv) {
  TORCH_INTERNAL_ASSERT(
      compute_with_tv->hasResolvedComputeWith(),
      "Invalid tensor: ",
      compute_with_tv->toString());

  id_graph_.updateComputeWith(compute_with_tv);

  // Update the LOOP concrete IDs
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdsSet(IdMappingMode::LOOP).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
