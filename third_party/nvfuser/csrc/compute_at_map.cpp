#include <compute_at_map.h>

#include <disjoint_set.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <lower_trivial_broadcast.h>
#include <lower_utils.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <tuple>
#include <typeinfo>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IterDomainGraph::IterDomainGraph(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool allow_self_mapping) {
  build(exprs, additional_tvs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

IterDomainGraph::IterDomainGraph(
    const std::vector<Expr*>& exprs,
    bool allow_self_mapping)
    : IterDomainGraph(exprs, {}, allow_self_mapping) {}

IterDomainGraph::IterDomainGraph(Fusion* fusion, bool allow_self_mapping) {
  std::vector<TensorView*> inputs_and_outputs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), inp_tvs.begin(), inp_tvs.end());
  }
  {
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), out_tvs.begin(), out_tvs.end());
  }

  build(fusion->exprs(), inputs_and_outputs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

const DisjointSets<IterDomain*>& IterDomainGraph::getDisjointIdSets(
    IdMappingMode mode) const {
  auto disjoint_ids_it = disjoint_ids_.find(mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_ids_it != disjoint_ids_.end(),
      "Mapping mode ",
      mode,
      " not supported.");
  return disjoint_ids_it->second;
}

std::pair<IdGroup, bool> IterDomainGraph::getDisjointIdSet(
    IterDomain* id,
    IdMappingMode mode) const {
  auto disjoint_mode_it = disjoint_ids_.find(mode);

  auto null_return = std::make_pair(IdGroup(nullptr), false);

  if (disjoint_mode_it == disjoint_ids_.end()) {
    return null_return;
  }

  const auto& disjoint_set = disjoint_mode_it->second;
  auto disjoint_set_it = disjoint_set.disjointSetMap().find(id);
  if (disjoint_set_it == disjoint_set.disjointSetMap().end()) {
    return null_return;
  }

  return std::make_pair(disjoint_set_it->second, true);
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

const DisjointSets<Expr*>& IterDomainGraph::getDisjointExprSets(
    IdMappingMode mode) const {
  auto disjoint_exprs_it = disjoint_exprs_.find(mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_exprs_it != disjoint_exprs_.end(),
      "Mapping mode ",
      mode,
      " not supported.");
  return disjoint_exprs_it->second;
}

std::pair<ExprGroup, bool> IterDomainGraph::getDisjointExprSet(
    Expr* expr,
    IdMappingMode mode) const {
  auto disjoint_mode_it = disjoint_exprs_.find(mode);

  auto null_return = std::make_pair(ExprGroup(nullptr), false);

  if (disjoint_mode_it == disjoint_exprs_.end()) {
    return null_return;
  }

  const auto& disjoint_set = disjoint_mode_it->second;
  auto disjoint_set_it = disjoint_set.disjointSetMap().find(expr);
  if (disjoint_set_it == disjoint_set.disjointSetMap().end()) {
    return null_return;
  }

  return std::make_pair(disjoint_set_it->second, true);
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

Expr* IterDomainGraph::idUse(IterDomain* id) const {
  auto use_it = id_uses_.find(id);
  if (use_it == id_uses_.end()) {
    return nullptr;
  }
  return use_it->second;
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
              return !getDisjointIdSets(mode).permissiveAreMapped(
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

  ExprGroups defs0;
  ExprGroups defs1;
  ExprGroups uses0;
  ExprGroups uses1;

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

void IterDomainGraph::assertNoSelfMapping() {
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
      if (id_graph.getDisjointIdSets(mode).permissiveAreMapped(id1, id2)) {
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
findFirstSelfMapping(
    const std::vector<TensorView*>& all_tvs,
    const IterDomainGraph& id_graph) {
  for (auto tv : all_tvs) {
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
  }

  if (is_view_rfactor_id) {
    view_rfactor_ids_.emplace(id);
  }
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
IterDomainGraph::buildMapBetween(
    const std::vector<IterDomain*>& from_ids,
    const std::vector<IterDomain*>& to_ids,
    IdMappingMode mode) const {
  std::unordered_map<IterDomain*, IdGroup> from_ids2set;

  for (auto from_id : from_ids) {
    auto from_disjoint_set_pair = getDisjointIdSet(from_id, mode);
    if (!from_disjoint_set_pair.second) {
      continue;
    }
    from_ids2set[from_id] = from_disjoint_set_pair.first;
  }

  // Map from the sets associated with the IterDomains in to, to those iter
  // domains
  std::unordered_map<IdGroup, VectorOfUniqueEntries<IterDomain*>> set2to_ids;

  for (auto to_id : to_ids) {
    auto to_disjoint_set_pair = getDisjointIdSet(to_id, mode);
    if (!to_disjoint_set_pair.second) {
      continue;
    }
    auto to_set = to_disjoint_set_pair.first;
    auto set2to_ids_it = set2to_ids.find(to_set);

    if (set2to_ids_it == set2to_ids.end()) {
      set2to_ids[to_set] = {to_id};
    } else {
      set2to_ids[to_set].pushBack(to_id);
    }
  }

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      from_ids2to_ids;
  for (auto from_id : from_ids) {
    from_ids2to_ids[from_id] = VectorOfUniqueEntries<IterDomain*>();

    auto from_it = from_ids2set.find(from_id);
    if (from_it == from_ids2set.end()) {
      continue;
    }

    auto from_set = from_it->second;
    auto to_entry_it = set2to_ids.find(from_set);
    if (to_entry_it == set2to_ids.end()) {
      continue;
    }
    from_ids2to_ids[from_id] = to_entry_it->second;
  }
  return from_ids2to_ids;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
IterDomainGraph::buildMapBetween(
    const VectorOfUniqueEntries<IterDomain*>& from_ids,
    const VectorOfUniqueEntries<IterDomain*>& to_ids,
    IdMappingMode mode) const {
  return buildMapBetween(from_ids.vector(), to_ids.vector(), mode);
}

std::pair<ExprGroups, bool> IterDomainGraph::getIterDomainGroupDefinitions(
    IdGroup id_group,
    IdMappingMode mode) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto mode_it = unique_definitions_.find(mode);
  if (mode_it == unique_definitions_.end()) {
    return null_return;
  }

  auto definition_it = mode_it->second.find(id_group);
  if (definition_it == mode_it->second.end()) {
    return null_return;
  }

  return std::make_pair(definition_it->second, true);
}

std::pair<ExprGroups, bool> IterDomainGraph::getIterDomainGroupUses(
    IdGroup id_group,
    IdMappingMode mode) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto mode_it = unique_uses_.find(mode);
  if (mode_it == unique_uses_.end()) {
    return null_return;
  }

  auto uses_it = mode_it->second.find(id_group);
  if (uses_it == mode_it->second.end()) {
    return null_return;
  }

  return std::make_pair(uses_it->second, true);
}

void IterDomainGraph::buildIterDomainUses(
    const std::vector<TensorView*>& all_tvs) {
  for (auto tv : all_tvs) {
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

// TODO: Extend to include other information.
std::string IterDomainGraph::toString() const {
  std::stringstream ss;
  ss << "IterDomainGraph { \n";
  for (auto set : disjoint_ids_) {
    ss << "Set " << set.first << ": " << std::endl;
    ss << set.second.toString() << std::endl;
  }
  ss << " } IterDomainGraph\n" << std::endl;
  return ss.str();
}

// Replay Expr but with the inputs provided. Input mapping will set a pairwise
// mapping between new_inputs and expr->inputs()
Expr* IterDomainGraph::addReplayAs(
    const std::vector<IterDomain*>& new_inputs,
    Expr* expr,
    IdMappingMode input_mapping) {
  std::vector<IdMappingMode> input_modes;
  switch (input_mapping) {
    case IdMappingMode::EXACT: {
      input_modes.push_back(IdMappingMode::EXACT);
      __attribute__((fallthrough));
    }
    case IdMappingMode::ALMOSTEXACT: {
      input_modes.push_back(IdMappingMode::ALMOSTEXACT);
      __attribute__((fallthrough));
    }
    case IdMappingMode::PERMISSIVE: {
      input_modes.push_back(IdMappingMode::PERMISSIVE);
      break;
    }
    case IdMappingMode::LOOP: {
      TORCH_INTERNAL_ASSERT(
          false,
          "Cannot replay transformations as input loop maps.",
          " Loop mappings have to be managed manually from TensorDomain leaves and compute at structure.");
    }
  }

  auto orig_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
  std::vector<IterDomain*> orig_input_ids(
      orig_inputs.begin(), orig_inputs.end());
  TORCH_INTERNAL_ASSERT(
      new_inputs.size() == orig_input_ids.size(),
      "Invalid number of inputs: ",
      new_inputs.size(),
      " does not match number of iter domain inputs for ",
      expr->toString());
  for (auto input_mode : input_modes) {
    for (auto inp_i : c10::irange(orig_input_ids.size())) {
      mapIds(orig_input_ids[inp_i], new_inputs[inp_i], input_mode);
    }
  }

  auto replay = ReplayTransform::replayAs(new_inputs, expr);

  for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
    initializeId(out_id, false, false);
    // This should be run after IterDomain graph is built, initializeId doesn't
    // initialize entries in the other maps.
    disjointIdsSet(IdMappingMode::ALMOSTEXACT).initializeSet(out_id);
    disjointIdsSet(IdMappingMode::PERMISSIVE).initializeSet(out_id);
  }

  // Propagate mappings from inputs
  mapThroughExpr(expr, replay, true, IdMappingMode::PERMISSIVE);

  ExprGroups all_uses;

  for (auto inp : orig_input_ids) {
    auto uses_pair = getIterDomainGroupUses(
        getDisjointIdSet(inp, IdMappingMode::PERMISSIVE).first,
        IdMappingMode::PERMISSIVE);
    if (uses_pair.second) {
      all_uses.pushBack(uses_pair.first);
    }
  }

  for (auto expr_set : all_uses) {
    auto first_expr = expr_set->front();
    // Simply try to map through the expressions, will only actually
    // happen if they map (exprsMap is checked in mapThroughExpr)
    mapThroughExpr(first_expr, replay, true, IdMappingMode::EXACT);
    mapThroughExpr(first_expr, replay, true, IdMappingMode::ALMOSTEXACT);
    mapThroughExpr(first_expr, replay, true, IdMappingMode::PERMISSIVE);
  }

  return replay;
}

void IterDomainGraph::initialIdProcessing(
    const std::vector<TensorView*>& all_tvs) {
  // Initialize entries for every iteration domain and mark view like
  // iteration domains and leaf iteration domains.
  for (auto tv : all_tvs) {
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

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    // Map siblings, as all other tv output domains must match the first tv
    // outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    for (auto other_tv_output : other_tv_outputs) {
      // Sibling tv's must be exactly mapped with eachother so simply zip their
      // leaf iter domains.

      TORCH_INTERNAL_ASSERT(
          other_tv_output->getRootDomain().size() ==
              c_tv->getRootDomain().size(),
          "Multiple outputs with mismatched TV domains is not supported.");

      for (auto domain_i : c10::irange(c_tv->getRootDomain().size())) {
        auto c_id = c_tv->getRootDomain()[domain_i];
        auto o_id = other_tv_output->getRootDomain()[domain_i];
        mapIds(o_id, c_id, IdMappingMode::EXACT);
      }
    }

    // Map producer-consumer relationships based on the root domain map
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
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    // Initialize all leaf nodes in loop id set
    for (auto tv_out : all_tv_outputs) {
      for (auto id : tv_out->domain()->domain()) {
        disjointIdsSet(IdMappingMode::LOOP).initializeSet(id);
      }
    }

    // Map siblings in loop map, as all other tv output domains must match the
    // first tv outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    for (auto other_tv_output : other_tv_outputs) {
      // Sibling tv's must be exactly mapped with eachother so simply zip their
      // leaf iter domains.
      TORCH_INTERNAL_ASSERT(
          other_tv_output->domain()->domain().size() ==
              c_tv->domain()->domain().size(),
          "Multiple outputs with mismatched TV domains is not supported.");

      for (auto domain_i : c10::irange(c_tv->domain()->domain().size())) {
        auto c_id = c_tv->domain()->domain()[domain_i];
        auto o_id = other_tv_output->domain()->domain()[domain_i];
        TORCH_INTERNAL_ASSERT(
            disjoint_ids_.at(IdMappingMode::EXACT).strictAreMapped(o_id, c_id),
            "Sibling domains must exact match however the following domains do not:\n  ",
            c_tv->toString(),
            "\n  ",
            other_tv_output->toString());
        mapIds(o_id, c_id, IdMappingMode::LOOP);
      }
    }

    // IterDomains from consumer that may match those in the producers
    std::vector<IterDomain*> c_ca_domain(
        c_tv->domain()->domain().begin(),
        c_tv->domain()->domain().begin() + c_tv->getMaxProducerPosition());

    if (c_ca_domain.empty()) {
      continue;
    }

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto p_tv : tv_inputs) {
      // Fusion inputs aren't involved in loop generation.
      if (p_tv->isFusionInput()) {
        continue;
      }

      // IterDomains from producer that may match with those in the first
      // consumer
      std::vector<IterDomain*> p_ca_domain(
          p_tv->domain()->domain().begin(),
          p_tv->domain()->domain().begin() + p_tv->getComputeAtPosition());

      // If producer is compute with the consumer, extend the matching domain to
      // the compute with of the producer.
      //
      // This shouldn't actually exist until after the compute at map is built
      // because it requires expression sorting to be run. To actually handle
      // this IterDomainGraph::updateComputeWith is being run after expression
      // sorting which can resolve the compute with of tensors.
      //
      // I'm leaving this in here as if we could resolve that before we build
      // the IterDomainGraph it's easy to handle here.
      if (p_tv->hasResolvedComputeWith()) {
        auto with_tvs = p_tv->getComputeWithConsumers();
        if (std::find(with_tvs.begin(), with_tvs.end(), c_tv) !=
            with_tvs.end()) {
          p_ca_domain = std::vector<IterDomain*>(
              p_tv->domain()->domain().begin(),
              p_tv->domain()->domain().begin() +
                  p_tv->getComputeWithPosition());
        }
      }

      if (p_ca_domain.empty()) {
        continue;
      }

      // Map densly in matching entries of consumer and producer domains.
      for (auto c_id_i : c10::irange(c_ca_domain.size())) {
        auto c_id = c_ca_domain[c_id_i];
        auto p_id_it = std::find_if(
            p_ca_domain.begin(), p_ca_domain.end(), [&](IterDomain* p_id) {
              return getDisjointIdSets(IdMappingMode::PERMISSIVE)
                  .permissiveAreMapped(c_id, p_id);
            });
        if (p_id_it != p_ca_domain.end()) {
          mapIds(c_id, *p_id_it, IdMappingMode::LOOP);
        }
      }
    }
  }
}

void IterDomainGraph::build(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs) {
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

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        TORCH_INTERNAL_ASSERT(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);
  if (additional_tvs.size() > 0) {
    std::unordered_set<TensorView*> all_added_tvs(
        all_tvs.begin(), all_tvs.end());
    for (auto additional_tv : additional_tvs) {
      if (all_added_tvs.find(additional_tv) == all_added_tvs.end()) {
        all_tvs.push_back(additional_tv);
      }
    }
  }

  if (all_tvs.empty()) {
    return;
  }

  FusionGuard fg(all_tvs.front()->fusion());

  // Add uses to all iter domains.
  buildIterDomainUses(all_tvs);

  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  initialIdProcessing(all_tvs);

  buildExactMap(tv_exprs);
  buildAlmostExactMap();
  buildPermissiveMap(tv_exprs);

  // Only build loop map during lowering
  if (FusionGuard::getCurFusion()->isA<kir::Kernel>()) {
    buildLoopMap(tv_exprs);

    // Find loops that need to be promoted because of broadcast resolution,
    // figure out what that resolution should look like, compute IDs for it if
    // necessary.
    buildLoopPromotionMap();
  }

  // Debug, make sure there's no self mapping in TensorView's during lowering
  // that would invalidate lowering assumptions.
  self_mapping_info_ = findFirstSelfMapping(all_tvs, *this);
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

      ExprGroups new_exprs;

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

namespace {

// Returns the root producer iteration domains that are resolved by provided
// consumer
std::unordered_map<IterDomain*, IterDomain*> resolvedRootBroadcasts(
    TensorView* producer,
    TensorView* consumer) {
  auto p2c_map =
      PairwiseRootDomainMap(producer, consumer)
          .mapProducerToConsumer(producer->domain(), consumer->domain());

  std::unordered_map<IterDomain*, IterDomain*> resolved_bcast_map;
  for (const auto& kv : p2c_map) {
    auto p_id = kv.first;
    // Ignore non-broadcast dims
    if (!p_id->isBroadcast()) {
      continue;
    }
    auto c_id = kv.second;
    // If the consumer ID is a reduction (i.e., a trivial
    // reduction), do not consider it's concretized.
    if (c_id->isBroadcast() || c_id->isReduction()) {
      continue;
    }
    resolved_bcast_map[p_id] = c_id;
  }
  return resolved_bcast_map;
}

} // namespace

ExprGroups IterDomainGraph::toGroups(
    const VectorOfUniqueEntries<Expr*>& exprs,
    IdMappingMode mode) const {
  ExprGroups groups;
  for (auto expr : exprs) {
    auto disjoint_set_pair = getDisjointExprSet(expr, mode);
    if (disjoint_set_pair.second) {
      groups.pushBack(disjoint_set_pair.first);
    }
  }
  return groups;
}

IdGroups IterDomainGraph::toGroups(
    const VectorOfUniqueEntries<IterDomain*>& ids,
    IdMappingMode mode) const {
  IdGroups groups;
  for (auto id : ids) {
    auto disjoint_set_pair = getDisjointIdSet(id, mode);
    if (disjoint_set_pair.second) {
      groups.pushBack(disjoint_set_pair.first);
    }
  }
  return groups;
}

IdGroups IterDomainGraph::outputGroups(ExprGroup expr, IdMappingMode mode)
    const {
  VectorOfUniqueEntries<IterDomain*> id_outputs;
  for (auto id_output :
       ir_utils::filterByType<IterDomain>(expr->front()->outputs())) {
    id_outputs.pushBack(id_output);
  }

  return toGroups(id_outputs, mode);
}

IdGroups IterDomainGraph::inputGroups(ExprGroup expr, IdMappingMode mode)
    const {
  VectorOfUniqueEntries<IterDomain*> id_inputs;
  for (auto id_input :
       ir_utils::filterByType<IterDomain>(expr->front()->inputs())) {
    id_inputs.pushBack(id_input);
  }

  return toGroups(id_inputs, mode);
}

ExprGroups IterDomainGraph::allUsesOf(const IdGroups& of, IdMappingMode mode)
    const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_uses_pair = getIterDomainGroupUses(of_id_group, mode);
    if (group_uses_pair.second) {
      to_visit.pushBack(group_uses_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto output_ids = outputGroups(current_expr, mode);
    for (auto output_id : output_ids) {
      auto group_uses_pair = getIterDomainGroupUses(output_id, mode);
      if (!group_uses_pair.second) {
        continue;
      }
      for (auto group_use : group_uses_pair.first) {
        if (visited.has(group_use)) {
          continue;
        }
        to_visit.pushBack(group_use);
      }
    }
  }

  return visited;
}

ExprGroups IterDomainGraph::allDefinitionsOf(
    const IdGroups& of,
    IdMappingMode mode) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_defs_pair = getIterDomainGroupDefinitions(of_id_group, mode);
    if (group_defs_pair.second) {
      to_visit.pushBack(group_defs_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto input_ids = inputGroups(current_expr, mode);
    for (auto input_id : input_ids) {
      auto group_defs_pair = getIterDomainGroupDefinitions(input_id, mode);
      if (!group_defs_pair.second) {
        continue;
      }
      for (auto group_def : group_defs_pair.first) {
        if (visited.has(group_def)) {
          continue;
        }
        to_visit.pushBack(group_def);
      }
    }
  }

  return visited;
}

// TODO: This seems really heavy weight, would be good to explore if there's
// better options here. It's called quite a bit in buildLoopPromotionMap
ExprGroups IterDomainGraph::getExprsBetween(
    const IdGroups& from,
    const IdGroups& to,
    IdMappingMode mode) const {
  auto all_uses_of_from = allUsesOf(from, mode);
  auto all_definitions_of_to = allDefinitionsOf(to, mode);

  // All of the expressions between from and to. Not all will be used as we just
  // want to define each iter domain group once.
  auto all_exprs = all_uses_of_from.intersect(all_definitions_of_to);

  // There could be IterDomains in from or to that are between other from and to
  // nodes. We should make sure to clear those out.
  IdGroups terminating_inputs;
  IdGroups terminating_outputs;
  {
    IdGroups not_inputs;
    IdGroups not_outputs;
    IdGroups all_id_groups;

    for (auto expr_group : all_exprs) {
      auto first_expr = expr_group->front();
      for (auto inp_id :
           ir_utils::filterByType<IterDomain>(first_expr->inputs())) {
        auto inp_group_pair = getDisjointIdSet(inp_id, mode);
        TORCH_INTERNAL_ASSERT(
            inp_group_pair.second,
            "Couldn't find group of required IterDomain.");
        auto inp_group = inp_group_pair.first;
        all_id_groups.pushBack(inp_group);
        not_outputs.pushBack(inp_group);
      }
      for (auto out_id :
           ir_utils::filterByType<IterDomain>(first_expr->outputs())) {
        auto out_group_pair = getDisjointIdSet(out_id, mode);
        TORCH_INTERNAL_ASSERT(
            out_group_pair.second,
            "Couldn't find group of required IterDomain.");
        auto out_group = out_group_pair.first;
        all_id_groups.pushBack(out_group);
        not_inputs.pushBack(out_group);
      }
    }
    terminating_inputs = all_id_groups.subtract(not_inputs);
    terminating_outputs = all_id_groups.subtract(not_outputs);
  }

  // Track all expressions to get from outputs to this IterDomain. We
  // traverse backwards as that's the direction of indexing expressions. An
  // index is assigned to each leaf of a domain and as we traverse backwards
  // we're effectively accumulating indexing math. We'll only keep the fewest
  // expression lists to get to the iter domain.
  std::unordered_map<IdGroup, ExprGroups> required_ind_exprs_ids;
  std::unordered_map<ExprGroup, ExprGroups> required_ind_exprs_exprs;

  // Return if all output IterDomain groups of an expression group have already
  // been visited
  auto outputsVisited = [&](ExprGroup expr) {
    for (auto id_group : outputGroups(expr, mode)) {
      if (required_ind_exprs_ids.find(id_group) ==
          required_ind_exprs_ids.end()) {
        return false;
      }
    }
    return true;
  };

  auto allIdUsesVisisted = [&](IdGroup id) {
    auto uses_pair = getIterDomainGroupUses(id, mode);
    if (!uses_pair.second) {
      return true;
    }
    for (auto use_group : uses_pair.first) {
      if (all_exprs.has(use_group)) {
        if (required_ind_exprs_exprs.find(use_group) ==
            required_ind_exprs_exprs.end()) {
          return false;
        }
      }
    }
    return true;
  };

  // Returns all expression groups in required_ind_exprs_ids of outputs
  auto requiredExprsOutputs = [&](ExprGroup expr) {
    ExprGroups all_output_required_exprs;
    for (auto id_group : outputGroups(expr, mode)) {
      auto id_group_exprs_it = required_ind_exprs_ids.find(id_group);
      TORCH_INTERNAL_ASSERT(
          id_group_exprs_it != required_ind_exprs_ids.end(),
          "Failure in Iter Domain Graph index resolution, count expected for group: ",
          id_group->toString());
      all_output_required_exprs.pushBack(id_group_exprs_it->second);
    }
    return all_output_required_exprs;
  };

  auto processExpr = [&](ExprGroup expr) {
    if (!outputsVisited(expr)) {
      return false;
    }
    // Accumulate expressions from all outputs add this expression and set it as
    // current expressions required indexing expressions.
    required_ind_exprs_exprs[expr] = requiredExprsOutputs(expr);
    return true;
  };

  auto processId = [&](IdGroup id) {
    // Track if we've grabed any of the uses required indexing expressions.
    bool initialized = false;
    // Expression group of all indexing expressions required for this iter
    // domain coming back from any of its uses.
    ExprGroups min_groups;

    auto uses_pair = getIterDomainGroupUses(id, mode);
    if (!uses_pair.second) {
      // No expressions required for this iter domain, it must be a
      // terminating output.
      required_ind_exprs_ids[id] = min_groups;
      return true;
    }

    // Only worry about expressions between inputs and outputs we're
    // looking at.
    for (auto use_group : uses_pair.first.intersect(all_exprs)) {
      auto use_required_ind_exprs_it = required_ind_exprs_exprs.find(use_group);
      if (use_required_ind_exprs_it == required_ind_exprs_exprs.end()) {
        // If there isn't an entry for the use expression it wasn't
        // processed, so don't try to process this iter domain yet.
        return false;
      }
      if (!initialized) {
        // If first use found initialize the minimum expression group
        min_groups =
            use_required_ind_exprs_it->second.computeUnion({use_group});
        initialized = true;
      } else if (
          use_required_ind_exprs_it->second.size() + 1 < min_groups.size()) {
        // If current use has fewer expressions use that, make sure to add the
        // use expression.
        min_groups =
            use_required_ind_exprs_it->second.computeUnion({use_group});
      }
    }
    required_ind_exprs_ids[id] = min_groups;
    return true;
  };

  IdGroups to_visit_ids = terminating_outputs;
  ExprGroups to_visit_exprs;

  while (to_visit_ids.size() > 0 || to_visit_exprs.size() > 0) {
    // Process expressions first as all uses of iter domains have to be
    // processed before we can process that iter domain.

    // Try to detect when nothing has been processed which would put us in an
    // infinite loop
    bool something_was_processed = false;
    ExprGroups still_to_visit_exprs;
    while (to_visit_exprs.size() > 0) {
      auto currently_visiting = to_visit_exprs.popFront();
      if (required_ind_exprs_exprs.find(currently_visiting) !=
          required_ind_exprs_exprs.end()) {
        continue;
      }
      if (processExpr(currently_visiting)) {
        something_was_processed = true;
        auto inp_groups = inputGroups(currently_visiting, mode);
        for (auto inp_group : inp_groups) {
          to_visit_ids.pushBack(inp_group);
        }
      } else {
        still_to_visit_exprs.pushBack(currently_visiting);
      }
    }

    std::swap(to_visit_exprs, still_to_visit_exprs);

    IdGroups still_to_visit_ids;
    while (to_visit_ids.size() > 0) {
      auto currently_visiting = to_visit_ids.popFront();
      if (required_ind_exprs_ids.find(currently_visiting) !=
          required_ind_exprs_ids.end()) {
        continue;
      }

      if (processId(currently_visiting)) {
        something_was_processed = true;
        auto definitions_pair =
            getIterDomainGroupDefinitions(currently_visiting, mode);
        if (definitions_pair.second) {
          for (auto def : definitions_pair.first) {
            if (!all_exprs.has(def)) {
            }
            if (required_ind_exprs_exprs.find(def) ==
                required_ind_exprs_exprs.end()) {
              to_visit_exprs.pushBack(def);
            }
          }
        }
      } else {
        still_to_visit_ids.pushBack(currently_visiting);
      }
    }

    TORCH_INTERNAL_ASSERT(
        something_was_processed ||
            (to_visit_ids.size() == 0 && to_visit_exprs.size() == 0),
        "Infinite loop entered.");
  }

  // We want to traverse the expressions registered in required_ind_exprs_ids,
  // let's create a strict "uses path"
  std::unordered_map<IdGroup, ExprGroups> uses_path;
  for (auto entry : required_ind_exprs_ids) {
    auto id = entry.first;
    auto traverse_exprs = entry.second;
    auto all_uses = getIterDomainGroupUses(id, mode);
    if (all_uses.second) {
      uses_path[id] = traverse_exprs.intersect(all_uses.first);
    } else {
      uses_path[id] = {};
      continue;
    }
  }

  // Topologically sort the uses_path.
  ExprGroups sorted_exprs;
  ExprGroups to_visit;

  for (auto inp : terminating_inputs) {
    auto use_it = uses_path.find(inp);
    TORCH_INTERNAL_ASSERT(
        use_it != uses_path.end(),
        "Invalid calculation of exprs between, no use found of terminating input: ",
        inp->toString());
    auto uses = use_it->second;
    for (auto use : uses) {
      to_visit.pushBack(use);
    }
  }

  IdGroups visited = terminating_inputs;

  while (to_visit.size() > 0) {
    bool something_processed = false;
    ExprGroups still_to_visit;
    while (to_visit.size() > 0) {
      auto currently_visiting = to_visit.popFront();
      auto inputs = inputGroups(currently_visiting, mode);
      if (std::all_of(inputs.begin(), inputs.end(), [&](IdGroup inp_id) {
            return visited.has(inp_id);
          })) {
        something_processed = true;
        sorted_exprs.pushBack(currently_visiting);
        auto outputs = outputGroups(currently_visiting, mode);
        for (auto out_id : outputs) {
          visited.pushBack(out_id);
          auto use_pair = getIterDomainGroupUses(out_id, mode);
          if (!use_pair.second) {
            continue;
          }
          still_to_visit.pushBack(use_pair.first.intersect(all_exprs));
        }
      } else {
        still_to_visit.pushBack(currently_visiting);
      }
    }
    std::swap(to_visit, still_to_visit);
    TORCH_INTERNAL_ASSERT(something_processed, "Infinite loop entered.");
  }

  return sorted_exprs;
}

void IterDomainGraph::buildLoopPromotionMap() {
  // Helper functions.
  auto producerIdGroups = [&](IdGroup id_group) {
    IdGroups producer_groups;
    auto definition_pair_it =
        getIterDomainGroupDefinitions(id_group, IdMappingMode::ALMOSTEXACT);
    if (!definition_pair_it.second) {
      return producer_groups;
    }
    for (auto def_group : definition_pair_it.first) {
      auto inp_groups = inputGroups(def_group, IdMappingMode::ALMOSTEXACT);
      producer_groups.pushBack(inp_groups);
    }
    return producer_groups;
  };

  auto consumerIdGroups = [&](IdGroup id_group) {
    IdGroups consumer_groups;
    auto uses_pair_it =
        getIterDomainGroupUses(id_group, IdMappingMode::ALMOSTEXACT);
    if (!uses_pair_it.second) {
      return consumer_groups;
    }
    for (auto use_group : uses_pair_it.first) {
      auto out_groups = outputGroups(use_group, IdMappingMode::ALMOSTEXACT);
      consumer_groups.pushBack(out_groups);
    }
    return consumer_groups;
  };

  auto all_tvs = ir_utils::allTvs(FusionGuard::getCurFusion());

  // Start at terminating inputs of the almost exact graph and almost exact
  // entries that are rfactor nodes. Propagate and accumulate these nodes
  // through consumers.
  //
  // The almost exact entries covered by an iteration domain is effectively all
  // the iteration domains this domain relies on. Initialize broadcast entries
  // to not cover any domains.
  std::unordered_map<IdGroup, IdGroups> covered_almost_exact_entries;

  // We will traverse over the almost exact set expressions. Save where we want
  // to start traversal:
  IdGroups to_visit;

  // Initialize traversal
  for (auto almost_exact_set :
       getDisjointIdSets(IdMappingMode::ALMOSTEXACT).disjointSets()) {
    // don't care what broadcast domains cover
    if (std::all_of(
            almost_exact_set->begin(),
            almost_exact_set->end(),
            [&](IterDomain* id) { return id->isBroadcast(); })) {
      covered_almost_exact_entries[almost_exact_set] = {};
      continue;
    }

    // Initialize rfactor domains to cover themselves only
    if (std::any_of(
            almost_exact_set->begin(),
            almost_exact_set->end(),
            [&](IterDomain* id) {
              return viewRfactorIds().find(id) != viewRfactorIds().end();
            })) {
      covered_almost_exact_entries[almost_exact_set] = {almost_exact_set};
      to_visit.pushBack(consumerIdGroups(almost_exact_set));
      continue;
    }

    // Initialize any groups that don't have a definition except (potentialy)
    // ones that traverse back to this set.
    auto def_pair = getIterDomainGroupDefinitions(
        almost_exact_set, IdMappingMode::ALMOSTEXACT);
    if (!def_pair.second) {
      covered_almost_exact_entries[almost_exact_set] = {almost_exact_set};
      to_visit.pushBack(consumerIdGroups(almost_exact_set));
      continue;
    }

    for (auto def : def_pair.first) {
      // If all definitions are self mapping (can happen with merging our
      // splitting with a broadcast/ dim of size 1) then this group is an input.
      auto inp_groups = inputGroups(def, IdMappingMode::ALMOSTEXACT);
      if (std::find(inp_groups.begin(), inp_groups.end(), almost_exact_set) ==
          inp_groups.end()) {
        goto loop_continue;
      }
    }

    covered_almost_exact_entries[almost_exact_set] = {almost_exact_set};
    to_visit.pushBack(consumerIdGroups(almost_exact_set));

  loop_continue:;
  }

  // Propagate covered ID groups
  while (to_visit.size() > 0) {
    IdGroups still_to_visit;
    bool something_processed = false;
    while (to_visit.size() > 0) {
      auto currently_visiting = to_visit.popFront();
      if (covered_almost_exact_entries.find(currently_visiting) !=
          covered_almost_exact_entries.end()) {
        continue;
      }
      auto producer_ids = producerIdGroups(currently_visiting);
      producer_ids.erase(currently_visiting);
      IdGroups currently_visiting_covered;
      for (auto producer_id : producer_ids) {
        auto producer_covered_it =
            covered_almost_exact_entries.find(producer_id);
        if (producer_covered_it == covered_almost_exact_entries.end()) {
          still_to_visit.pushBack(currently_visiting);
          goto inner_while_continue;
        }
        for (auto entry : producer_covered_it->second) {
          if (currently_visiting_covered.has(entry)) {
            continue;
          }
        }
        currently_visiting_covered.pushBack(producer_covered_it->second);
      }
      covered_almost_exact_entries[currently_visiting] =
          currently_visiting_covered;
      to_visit.pushBack(consumerIdGroups(currently_visiting));
      something_processed = true;

    inner_while_continue:;
    }
    TORCH_INTERNAL_ASSERT(
        still_to_visit.empty() || something_processed,
        "Entered infinite loop.");
    std::swap(still_to_visit, to_visit);
  }

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_ca_root_broadcast_resolution_map;

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_ca_permissive_maps;

  // Want to traverse the iter domains when we do promotion in topological
  // order, so we will save that ordering as we populate the above maps.
  VectorOfUniqueEntries<IterDomain*> ordered_p_ca_ids;

  for (auto producer : all_tvs) {
    auto producer_root = producer->getMaybeRFactorDomain();
    auto producer_domain = producer->domain()->domain();

    // Grab all iteration domains in producer that its compute at iter domains
    // depend on.
    VectorOfUniqueEntries<IterDomain*> all_producer_ca_deps;
    {
      auto ca_dep_vals = DependencyCheck::getAllValsBetween(
          {producer_root.begin(), producer_root.end()},
          {producer_domain.begin(),
           producer_domain.begin() + producer->getComputeAtPosition()});

      auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);

      all_producer_ca_deps.insert(ca_deps_filter.begin(), ca_deps_filter.end());
    }

    ordered_p_ca_ids.pushBack(all_producer_ca_deps);

    // Grab all iteration domains in producer between its compute at and max
    // produce at position depend on.
    VectorOfUniqueEntries<IterDomain*> all_producer_pa_deps;
    if (producer->getMaxProducerPosition() > producer->getComputeAtPosition()) {
      auto pa_dep_vals = DependencyCheck::getAllValsBetween(
          {producer_root.begin(), producer_root.end()},
          {producer_domain.begin() + producer->getComputeAtPosition(),
           producer_domain.begin() + producer->getMaxProducerPosition()});

      auto pa_deps_filter = ir_utils::filterByType<IterDomain>(pa_dep_vals);

      all_producer_pa_deps.insert(pa_deps_filter.begin(), pa_deps_filter.end());
    }

    // If provided map already has entry for key, accumulate into that entry the
    // new provided value.
    auto accumulateInMap =
        [](std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
               map,
           IterDomain* key,
           IterDomain* new_value) {
          auto entry_it = map.find(key);
          if (map.find(key) == map.end()) {
            map[key] = {new_value};
          } else {
            auto& value = entry_it->second;
            value.pushBack(new_value);
          }
        };

    auto consumers = ir_utils::consumerTvsOf(producer);
    for (auto consumer : consumers) {
      auto resolved_bcast_map = resolvedRootBroadcasts(producer, consumer);
      for (auto entry : resolved_bcast_map) {
        if (all_producer_ca_deps.has(entry.first) ||
            all_producer_pa_deps.has(entry.first)) {
          accumulateInMap(
              p2c_ca_root_broadcast_resolution_map, entry.first, entry.second);
          for (auto other_exact_bcast :
               *getDisjointIdSet(entry.first, IdMappingMode::EXACT).first) {
            if (all_producer_ca_deps.has(other_exact_bcast) ||
                all_producer_pa_deps.has(other_exact_bcast)) {
              accumulateInMap(
                  p2c_ca_root_broadcast_resolution_map,
                  other_exact_bcast,
                  entry.second);
            }
          }
        }
      }

      auto p2c_permissive_map = buildMapBetween(
          all_producer_ca_deps.vector(),
          ir_utils::allIDsOf(consumer),
          IdMappingMode::PERMISSIVE);

      for (auto entry : p2c_permissive_map) {
        // TODO: Should this be an assert instead of continue?
        if (entry.second.size() == 0) {
          continue;
        }

        accumulateInMap(
            p2c_ca_permissive_maps, entry.first, entry.second.back());
      }
    }
  }

  DisjointSets<IterDomain*> promotion_sets;
  for (auto entry : p2c_ca_permissive_maps) {
    auto first = entry.first;
    for (auto second : entry.second) {
      promotion_sets.mapEntries(first, second);
    }
  }

  // Even if there's no promotion, put into a promotion set.
  for (auto id : ordered_p_ca_ids) {
    promotion_sets.initializeSet(id);
  }

  for (auto set : promotion_sets.disjointSets()) {
    IdGroups to_cover;
    for (auto entry : *set) {
      if (p2c_ca_permissive_maps.find(entry) == p2c_ca_permissive_maps.end()) {
        to_cover.pushBack(covered_almost_exact_entries.at(
            getDisjointIdSet(entry, IdMappingMode::ALMOSTEXACT).first));
      }
    }
  }

  // Promotion map keys are the sets that share a promotion, these input sets
  // can be across permissive mapping.
  std::unordered_map<IdGroup, IterDomain*> promotion_map;

  IdGroups promote_groups;

  // Exact groups in promote_Groups
  IdGroups exact_groups_in_promote;

  // TODO: Order doesn't matter because we don't reuse anything in the
  // promotion computation. We should fix this see comment in computing the
  // promoted ID.
  for (auto promote_id : ordered_p_ca_ids) {
    auto promoted_id_it = promotion_sets.disjointSetMap().find(promote_id);
    TORCH_INTERNAL_ASSERT(
        promoted_id_it != promotion_sets.disjointSetMap().end(),
        promote_id->toString(),
        " not found in promotion map.");
    promote_groups.pushBack(promoted_id_it->second);
  }

  // Mark what's been promoted. When we search for expressions, no point in
  // going past what's already been promoted.
  IdGroups promoted_groups;

  // Working with three types of disjoint sets now, need to be careful how
  // they're mixed.
  // Promotion sets are defined based on groups that are share edges in the
  //   promotion map. They should all be promoted to the same type. They are
  //   permissive mapped by definition, but not necessarily almost or exact
  //   mapped.
  // AlmostExact mapping is used to see what iter domains need to be covered by
  //   the replay to cover a full promotion set. We don't need to cover every
  //   exact set in the history, but definitely need to cover all almost exact
  //   sets.
  // Exact mapping is used to perform the actual replay required to cover a full
  //   promotion set. If we have something like (7 * 1) and (1 * 13) the almost
  //   exact map might view these as 7 and 13 without the broadcast merge. We
  //   need the broadcast merge because we need to replay one of those.

  for (auto promote_group : promote_groups) {
    IdGroups to_cover;
    IdGroups terminal_ids;

    IdGroups exact_groups_in_promote_group;
    for (auto id : *promote_group) {
      exact_groups_in_promote_group.pushBack(
          getDisjointIdSet(id, IdMappingMode::EXACT).first);
    }

    // Group already found.
    if (promotion_map.find(promote_group) != promotion_map.end()) {
      continue;
    }

    for (auto entry : *promote_group) {
      if (p2c_ca_permissive_maps.find(entry) == p2c_ca_permissive_maps.end()) {
        // Careful, mixing modes in this analysis. EXACT is good to reproduce
        // transformations for this resolution. However, once promotion that
        // promotion can be shared on almost exact.
        auto exact_group_pair = getDisjointIdSet(entry, IdMappingMode::EXACT);
        TORCH_INTERNAL_ASSERT(exact_group_pair.second);
        terminal_ids.pushBack(exact_group_pair.first);
        auto almost_exact_group_pair =
            getDisjointIdSet(entry, IdMappingMode::ALMOSTEXACT);
        TORCH_INTERNAL_ASSERT(almost_exact_group_pair.second);
        to_cover.pushBack(
            covered_almost_exact_entries.at(almost_exact_group_pair.first));
      }
    }

    if (terminal_ids.size() == 1) {
      auto promoted_id = terminal_ids.front()->front();
      promotion_map[promote_group] = promoted_id;
      continue;
    }

    // Initialize early due to the goto used.
    bool promotion_found = false;

    for (auto terminal_id : terminal_ids) {
      // Almost exact should be a super set of exact which is where the
      // terminal_id is placed
      auto almost_exact_terminal_pair =
          getDisjointIdSet(terminal_id->front(), IdMappingMode::ALMOSTEXACT);
      TORCH_INTERNAL_ASSERT(almost_exact_terminal_pair.second);
      if (to_cover
              .subtract(covered_almost_exact_entries.at(
                  almost_exact_terminal_pair.first))
              .empty()) {
        promotion_map[promote_group] = terminal_id->front();
        promotion_found = true;
        break;
      }
    }

    if (promotion_found) {
      continue;
    }

    std::unordered_map<IdGroup, IdGroup> bcast_promotion_map;
    for (auto entry : p2c_ca_root_broadcast_resolution_map) {
      auto from = entry.first;
      auto tos = entry.second;
      for (auto to : tos) {
        if (to_cover.has(
                getDisjointIdSet(to, IdMappingMode::ALMOSTEXACT).first)) {
          // TODO: Make sure we're not trying to broadcast the same thing to two
          // different extents.
          bcast_promotion_map[getDisjointIdSet(from, IdMappingMode::EXACT)
                                  .first] =
              getDisjointIdSet(to, IdMappingMode::EXACT).first;
        }
      }
    }

    // A new IterDomain has to be created because none of the terminal_ids have
    // all the required covered IterDomains. Generate a new IterDomain that
    // satisfies the requirement of covering all of the almost exact sets in
    // "to_cover".

    // Compute all inputs we need to use to replay the terminal ids, start at
    // terminal ids and propagate backwards. Stop at iter domains that don't
    // require promotion, or those already promoted.
    std::unordered_map<IdGroup, IdGroup> local_promotion_map;

    IdGroups start_point;
    for (auto group : to_cover) {
      for (auto id : *group) {
        start_point.pushBack(getDisjointIdSet(id, IdMappingMode::EXACT).first);
      }
    }

    for (auto bcast_promo : bcast_promotion_map) {
      start_point.pushBack(bcast_promo.first);
    }

    auto all_exprs =
        getExprsBetween(start_point, terminal_ids, IdMappingMode::EXACT);

    // This replay has really bad complexity. Think about having IterDomains
    // that are dependent on eachother:
    //
    // ceilDiv(ceilDiv((7 * 1) * 13, 5), 3)
    //
    // Let's say this is a terminal ID and 1 needs to be broadcasted, we have:
    // 7 * 1
    // (7 * 1) * 13
    // ceilDiv((7 * 1) * 13, 5)
    // ceilDiv(ceilDiv((7 * 1) * 13, 5), 3)
    //
    // So we should only have to replay 4 times. However, this algorithm will
    // replay all previous expressions for all expressions. It will not reuse
    // the computations. Since 5 and 3 are also split off, full replays will be
    // performed for them too.
    //
    // Finding what we can reuse is a bit challenging. We should be able to
    // reuse iter domains that are promoted, and not replay all the way back
    // from inputs. However, I'm not sure if finding where we can start
    // traversal from is easy. We have a local_promotion_map that is not the
    // global_promotion_map. I don't believe these are the same in all cases.
    //
    // Leaving the bad complexity here for now, but should revisit and fix as
    // this could blow up quickly.

    for (auto expr : all_exprs) {
      std::vector<IterDomain*> new_input_ids;
      for (auto inp_group : inputGroups(expr, IdMappingMode::EXACT)) {
        auto bcast_promo_it = bcast_promotion_map.find(inp_group);
        if (bcast_promo_it != bcast_promotion_map.end()) {
          new_input_ids.push_back(bcast_promo_it->second->front());
          continue;
        }
        auto local_promo_it = local_promotion_map.find(inp_group);
        if (local_promo_it != local_promotion_map.end()) {
          new_input_ids.push_back(local_promo_it->second->front());
          continue;
        }

        new_input_ids.push_back(inp_group->front());
      }

      auto replayed_expr =
          addReplayAs(new_input_ids, expr->front(), IdMappingMode::PERMISSIVE);

      // A vector type would be nice.
      auto orig_outputs_ids =
          ir_utils::filterByType<IterDomain>(expr->front()->outputs());
      std::vector<IterDomain*> orig_outputs_ids_vec{
          orig_outputs_ids.begin(), orig_outputs_ids.end()};

      auto new_outputs_ids =
          ir_utils::filterByType<IterDomain>(replayed_expr->outputs());
      std::vector<IterDomain*> new_outputs_ids_vec{
          new_outputs_ids.begin(), new_outputs_ids.end()};

      TORCH_INTERNAL_ASSERT(
          orig_outputs_ids_vec.size() == new_outputs_ids_vec.size());

      // Add outputs to promotion map
      for (auto id_i : c10::irange(orig_outputs_ids_vec.size())) {
        auto orig_set_pair =
            getDisjointIdSet(orig_outputs_ids_vec[id_i], IdMappingMode::EXACT);
        auto replay_set_pair =
            getDisjointIdSet(new_outputs_ids_vec[id_i], IdMappingMode::EXACT);
        TORCH_INTERNAL_ASSERT(orig_set_pair.second && replay_set_pair.second);
        local_promotion_map[orig_set_pair.first] = replay_set_pair.first;
      }
    }

    for (auto terminal_id : terminal_ids) {
      if (local_promotion_map.find(terminal_id) != local_promotion_map.end()) {
        promotion_map[promote_group] =
            local_promotion_map.at(terminal_id)->front();
        promotion_found = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        promotion_found,
        "Error computing promoted iter domain for group: ",
        promote_group->toString());

    promoted_groups.pushBack(promote_group);
  }

  // Let's convert this to be on an IterDomain by IterDomain basis
  std::unordered_map<IterDomain*, IterDomain*> id_promotion_map;

  for (auto promotion_map_entry : promotion_map) {
    for (auto from_id : *promotion_map_entry.first) {
      auto to_id = promotion_map_entry.second;
      if (!getDisjointIdSets(IdMappingMode::ALMOSTEXACT)
               .permissiveAreMapped(from_id, to_id)) {
        id_promotion_map[from_id] = to_id;
      }
    }
  }

  // All promotions are done for shared loop nests, however we need to propagate
  // intermediate promotions to resolve dependencies outside shared loop nests.
  for (auto tv : all_tvs) {
    auto shared_loop_pos =
        std::max(tv->getMaxProducerPosition(), tv->getComputeAtPosition());
    if (tv->nDims() == shared_loop_pos || shared_loop_pos == 0) {
      // No leaf promotions needed, don't process
      continue;
    }

    auto domain = tv->domain()->domain();
    auto root = tv->getMaybeRFactorDomain();

    VectorOfUniqueEntries<IterDomain*> all_tv_ca_deps;
    {
      auto ca_dep_vals = DependencyCheck::getAllValsBetween(
          {root.begin(), root.end()},
          {domain.begin(), domain.begin() + shared_loop_pos});

      auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);

      all_tv_ca_deps.insert(ca_deps_filter.begin(), ca_deps_filter.end());
    }

    auto& all_promoted_ca_deps = all_tv_ca_deps;

    for (auto id : all_tv_ca_deps) {
      auto promoted_entry_it = id_promotion_map.find(id);
      if (promoted_entry_it == id_promotion_map.end()) {
        all_promoted_ca_deps.erase(id);
        continue;
      }

      auto promoted_id = promoted_entry_it->second;
      if (getDisjointIdSets(IdMappingMode::ALMOSTEXACT)
              .permissiveAreMapped(promoted_id, id)) {
        continue;
      }

      id_promotion_map[id] = promoted_id;
    }

    auto exprs = StmtSort::getExprsBetween(
        FusionGuard::getCurFusion(),
        {all_promoted_ca_deps.begin(), all_promoted_ca_deps.end()},
        {domain.begin() + tv->getComputeAtPosition(),
         domain.begin() + tv->nDims()});

    for (auto expr : exprs) {
      auto id_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
      std::vector<IterDomain*> input_copy{id_inputs.begin(), id_inputs.end()};

      bool input_promoted = false;

      for (auto input_i : c10::irange(input_copy.size())) {
        auto promote_it = id_promotion_map.find(input_copy[input_i]);

        if (promote_it == id_promotion_map.end()) {
          continue;
        }

        input_promoted = true;

        input_copy[input_i] = promote_it->second;
      }

      if (!input_promoted) {
        continue;
      }

      auto replay = addReplayAs(input_copy, expr, IdMappingMode::PERMISSIVE);

      // A vector type would be nice.
      auto orig_outputs_ids =
          ir_utils::filterByType<IterDomain>(expr->outputs());
      std::vector<IterDomain*> orig_outputs_ids_vec{
          orig_outputs_ids.begin(), orig_outputs_ids.end()};

      auto new_outputs_ids =
          ir_utils::filterByType<IterDomain>(replay->outputs());
      std::vector<IterDomain*> new_outputs_ids_vec{
          new_outputs_ids.begin(), new_outputs_ids.end()};

      TORCH_INTERNAL_ASSERT(
          orig_outputs_ids_vec.size() == new_outputs_ids_vec.size());

      // Add outputs to promotion map
      for (auto id_i : c10::irange(orig_outputs_ids_vec.size())) {
        id_promotion_map[orig_outputs_ids_vec[id_i]] =
            new_outputs_ids_vec[id_i];
      }
    }
  }

  // Make a copy as loop goups may change as we update them
  IdGroups loop_groups{
      disjointIdsSet(IdMappingMode::LOOP).disjointSets().begin(),
      disjointIdsSet(IdMappingMode::LOOP).disjointSets().end()};

  // There's an implicit assumption that loop id's only match if within the same
  // loop group. If a promoted id was already used we'll just copy it and map it
  // exact, almost exact, and permissive.

  VectorOfUniqueEntries<IterDomain*> used_loop_ids;
  for (auto loop_group : loop_groups) {
    for (auto id : *loop_group) {
      auto promoted_id_it = id_promotion_map.find(id);
      if (promoted_id_it == id_promotion_map.end()) {
        continue;
      }

      auto promoted_id = promoted_id_it->second;
      auto promoted_id_loop_group =
          getDisjointIdSet(promoted_id, IdMappingMode::LOOP);

      auto cloneAndMap = [&]() {
        auto new_promoted_id = IterDomainBuilder(promoted_id).build();
        mapIds(id, new_promoted_id, IdMappingMode::EXACT);
        mapIds(id, new_promoted_id, IdMappingMode::ALMOSTEXACT);
        mapIds(id, new_promoted_id, IdMappingMode::PERMISSIVE);
        mapIds(id, new_promoted_id, IdMappingMode::LOOP);
      };

      if (promoted_id_loop_group.second) {
        if (promoted_id_loop_group.first == loop_group) {
          // Already in the right loop group
          used_loop_ids.pushBack(promoted_id);
        } else {
          // In a different loop group, clone.
          cloneAndMap();
        }
      } else {
        if (used_loop_ids.has(promoted_id)) {
          cloneAndMap();
        } else {
          mapIds(id, promoted_id, IdMappingMode::LOOP);
          used_loop_ids.pushBack(promoted_id);
        }
      }
    }
  }
}

ComputeAtMap::ComputeAtMap(Fusion* fusion)
    : id_graph_(fusion), concretized_bcasts_(fusion), fusion_(fusion) {
  build(fusion);
}

void ComputeAtMap::build(Fusion* fusion) {
  buildConsumersMap();
  buildConcreteIds();
  testValidate();
}

// TODO: Cleanup, edges are unique expr's and nodes are disjoint sets
bool ComputeAtMap::indexingReachableFrom(
    const VectorOfUniqueEntries<IterDomain*>& from,
    const VectorOfUniqueEntries<IterDomain*>& to) {
  // Convert inputs to exact disjoint sets
  std::deque<IdGroup> to_visit;
  for (auto from_id : from) {
    to_visit.push_back(disjointSetOf(from_id, IdMappingMode::ALMOSTEXACT));
  }

  // Convert outputs to exact disjoint sets
  std::unordered_set<IdGroup> to_resolve;
  for (auto to_id : to) {
    to_resolve.emplace(disjointSetOf(to_id, IdMappingMode::ALMOSTEXACT));
  }

  // Any output that's also an input is automatically resolved remove them
  for (auto entry : to_visit) {
    to_resolve.erase(entry);
  }

  std::unordered_set<IdGroup> visited;
  visited.insert(to_visit.begin(), to_visit.end());

  // Collect nodes if we can't process them in not_visited, if we end up
  // visiting any node then add all not_visited to visited.
  //
  // That way if we have a case where we can't get from outputs to inputs,
  // not_visited will fill up as to_visit is being drained, signally we can't
  // make forward progress.
  //
  // Traversal is "backwards" so in_id's is actually expr->output
  // and out_id is actually expr->input
  std::deque<IdGroup> not_visited;
  while (!to_visit.empty() && !to_resolve.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();

    auto defs_it = id_graph_.getIterDomainGroupDefinitions(
        currently_visiting, IdMappingMode::ALMOSTEXACT);
    if (!defs_it.second) {
      TORCH_INTERNAL_ASSERT(
          currently_visiting->front()->definition() == nullptr,
          "unique_definitions_.at(IdMappingMode::ALMOSTEXACT) wasn't correctly generated, missing the disjoint set:\n",
          currently_visiting->toString());
    }

    // does not return one def, but multiple unique groups of exact defs.
    std::vector<Expr*> def_exprs;
    for (auto group : defs_it.first) {
      if (group->size() > 0) {
        def_exprs.push_back(group->front());
      }
    }

    {
      // Clear out any expression that's already been resolved
      decltype(def_exprs) unresolved_exprs;
      std::copy_if(
          def_exprs.begin(),
          def_exprs.end(),
          std::back_inserter(unresolved_exprs),
          [&](Expr* def_expr) {
            auto out_ids =
                ir_utils::filterByType<IterDomain>(def_expr->inputs());
            return std::any_of(
                out_ids.begin(), out_ids.end(), [&](IterDomain* out_id) {
                  return visited.find(disjointSetOf(
                             out_id, IdMappingMode::ALMOSTEXACT)) ==
                      visited.end();
                  // If any expression input has not been traversed we still
                  // can traverse def_expr
                });
          });

      std::swap(def_exprs, unresolved_exprs);
    }

    if (def_exprs.empty()) {
      // Nothing to resolve based on this set, just continue.
      continue;
    }

    // check if all def expressions have been resolved
    for (auto def_expr : def_exprs) {
      auto in_ids = ir_utils::filterByType<IterDomain>(def_expr->outputs());
      if (std::any_of(in_ids.begin(), in_ids.end(), [&](IterDomain* in_id) {
            return visited.find(disjointSetOf(
                       in_id, IdMappingMode::ALMOSTEXACT)) == visited.end();
          })) {
        // Cannot process this def_expr, continue all of the expr output ids
        // haven't been visited
        continue;
      }

      // All expr outputs were already visited, can mark this set as visited
      // and add expr inputs to to_visit
      // Visit nodes
      visited.emplace(currently_visiting);
      to_resolve.erase(currently_visiting);
      auto out_ids = ir_utils::filterByType<IterDomain>(def_expr->inputs());
      for (auto out_id : out_ids) {
        visited.emplace(disjointSetOf(out_id, IdMappingMode::ALMOSTEXACT));
        to_resolve.erase(disjointSetOf(out_id, IdMappingMode::ALMOSTEXACT));
      }

      // Move not_visited to back of to_visit as it may now be visitable
      to_visit.insert(to_visit.end(), not_visited.begin(), not_visited.end());
      not_visited.clear();

      // Add inputs to to_visit
      auto inp_ids = ir_utils::filterByType<IterDomain>(def_expr->inputs());
      for (auto inp_id : inp_ids) {
        to_visit.push_back(disjointSetOf(inp_id, IdMappingMode::ALMOSTEXACT));
      }
    }
  }

  if (!to_resolve.empty()) {
    std::cerr
        << "New indexing approach does not work here yet, did not resolve:"
        << std::endl;
    for (auto entry : to_resolve) {
      std::cerr << "  " << entry->toString() << std::endl;
    }
  }

  return to_resolve.empty();
}

void ComputeAtMap::testValidate() {
  // Scheduling can use compute at map, and may be in a bad state, only check
  // during lowering
  if (!FusionGuard::getCurFusion()->isA<kir::Kernel>()) {
    return;
  }

  auto all_tvs = ir_utils::allTvs(FusionGuard::getCurFusion());
  for (auto tv : all_tvs) {
    // Fusion inputs don't result in control flow, ignore.
    if (tv->isFusionInput()) {
      continue;
    }

    for (auto tv : all_tvs) {
      IdGroups tv_loop_domains;

      // Grab the iter domains that should be used for the for loops.
      VectorOfUniqueEntries<IterDomain*> loop_ids;
      for (auto id : tv->domain()->domain()) {
        // Traverse the promotion map until a leaf is found
        IterDomain* promoted_id = id_graph_.getMaybePromoted(id);

        while (promoted_id != id_graph_.getMaybePromoted(promoted_id)) {
          promoted_id = id_graph_.getMaybePromoted(promoted_id);
        }

        TORCH_INTERNAL_ASSERT(
            id_graph_.getDisjointIdSets(IdMappingMode::LOOP)
                .mappingExists(promoted_id),
            "Loop id's aren't inclusive, as a producer could look to promote to an IterDomain that's not a consumer's leaf domain.",
            " Error from trying to promote ",
            id,
            " to ",
            promoted_id);
        auto promoted_loop_concrete_id =
            getConcreteMappedID(promoted_id, IdMappingMode::LOOP);

        loop_ids.pushBack(promoted_loop_concrete_id);
      }

      // Grab the iter domains we need to index into
      VectorOfUniqueEntries<IterDomain*> root_ids;
      for (auto id : tv->getMaybeRFactorDomain()) {
        if (id->isBroadcast()) {
          // Broadcast IDs don't need to be indexable
          continue;
        }
        root_ids.pushBack(id);
      }

      // // TODO: Add assert once full loop promotion is implemented.
      // // Check if root is indexable based on loops
      // TORCH_INTERNAL_ASSERT(
      //     indexingReachableFrom(loop_ids, root_ids),
      //     "Could not detect how to resolve the indexing from loop
      //     IterDomains: ", loop_ids.toString(), " to root iter domains: ",
      //     root_ids.toString(),
      //     "\n When checking the indexing of ",
      //     tv->toString());
    }
  }
}

void ComputeAtMap::validateAndPropagatePType() {
  for (const auto& loop_disjoint_set :
       id_graph_.getDisjointIdSets(IdMappingMode::LOOP).disjointSets()) {
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
       id_graph_.getDisjointIdSets(IdMappingMode::LOOP).disjointSets()) {
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
      id_graph_.getDisjointIdSets(IdMappingMode::LOOP).mappingExists(id),
      "Index Variable: no index variable allocated as ",
      id->toString(),
      " is not registered in loop map");
  const auto* loop_set =
      id_graph_.getDisjointIdSet(id, IdMappingMode::LOOP).first.get();

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

  // Store set to the id that's actually in the disjoint set we're looking at.
  // This is only important for the loop concerete id detection as we want to
  // make sure what we return is in the loop disjoint set.
  std::unordered_map<IdGroup, IterDomain*> maybe_concrete_to_id;

  // Grab a set of candidate concrete_ids, we track towards the consumers in
  // the ID group as one of those is guaranteed to be a valid concrete id.
  IdGroups maybe_concrete_ids;
  for (auto disjoint_id : disjoint_set_shared_ptr->vector()) {
    bool id_output = true;
    auto consumers_it = consumers_map_.find(disjoint_id);
    if (consumers_it != consumers_map_.end()) {
      for (auto consumer_id : consumers_it->second.vector()) {
        if (disjoint_set_shared_ptr->has(consumer_id)) {
          id_output = false;
          break;
        }
      }
    }
    if (id_output) {
      auto disjoint_set_pair =
          id_graph_.getDisjointIdSet(disjoint_id, IdMappingMode::EXACT);
      TORCH_INTERNAL_ASSERT(disjoint_set_pair.second);
      maybe_concrete_to_id[disjoint_set_pair.first] = disjoint_id;
      maybe_concrete_ids.pushBack(disjoint_set_pair.first);
    }
  }

  // Shouldn't ever happen, it would mean there's an error somewhere in the
  // graph.
  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.size() > 0,
      "No potential concrete_id's found for ",
      id->toString());

  if (maybe_concrete_ids.size() == 1) {
    return maybe_concrete_to_id.at(maybe_concrete_ids.front());
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

  // Going to iteratively modify this to be all sets that the concrete ID
  // needs to cover
  IdGroups all_exact_sets_covered =
      getAllDisjointSetProducers(maybe_concrete_ids);

  // Remove all broadcast domains that are resolved within the history of any
  // of the maybe concrete sets.
  {
    // All broadcast exact sets in all_exact_sets_covered that are resolved by
    // IterDomains in all_exact_sets_covered
    IdGroups resolved_broadcasts;

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

    all_exact_sets_covered =
        all_exact_sets_covered.subtract(all_resolved_broadcast_uses);
  }

  // Remove all domains in the history of sets marked as rfactor.
  {
    // All exact sets in the history of an rfactored domain
    IdGroups produces_rfactor_dom;
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
      IdGroups rfactor_history = getAllDisjointSetProducers({exact_set});
      for (auto entry : rfactor_history) {
        // Leave rfactor exact set, unless it's in the history of another
        // rfactor domain.
        if (entry != exact_set) {
          produces_rfactor_dom.pushBack(entry);
        }
      }
    }

    // Remove all sets in rfactor history from all_exact_sets_covered
    all_exact_sets_covered =
        all_exact_sets_covered.subtract(produces_rfactor_dom);
  }

  maybe_concrete_ids = maybe_concrete_ids.intersect(all_exact_sets_covered);

  IdGroups input_ids;

  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.size() > 0,
      "No potential concrete_id's found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  if (maybe_concrete_ids.size() == 1) {
    return maybe_concrete_to_id.at(maybe_concrete_ids.front());
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
        [&](IdGroup set) { return set->vector()[0]->isBroadcast(); });

    int iter_root_count = (int)concrete_id_root_sets.size() - bcast_root_count;
    if (iter_root_count > max_iter_root_count ||
        (iter_root_count == max_iter_root_count &&
         bcast_root_count > max_bcast_root_count)) {
      max_iter_root_count = iter_root_count;
      max_bcast_root_count = bcast_root_count;
      concrete_id = maybe_concrete_to_id.at(maybe_concrete_id);
    }
  }

  TORCH_INTERNAL_ASSERT(
      concrete_id != nullptr,
      "No concrete_id found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  return concrete_id;
}

void ComputeAtMap::buildConsumersMap() {
  // To build concrete maps we will need to know the consumers of the
  // IterDomains in the permissive map. Build this map.

  // Filter non-TensorView expressions
  auto all_exprs = fusion_->exprs();
  std::vector<Expr*> tv_exprs;

  std::copy_if(
      all_exprs.begin(),
      all_exprs.end(),
      std::back_inserter(tv_exprs),
      [](Expr* expr) { return ir_utils::isTvOp(expr); });

  for (auto expr : tv_exprs) {
    auto consumers = ir_utils::filterByType<TensorView>(expr->outputs());
    auto producers = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto consumer : consumers) {
      auto all_consumer_ids = ir_utils::allIDsOf(consumer);
      // Change data structure for IterDomainGraph::buildMapBetween
      VectorOfUniqueEntries<IterDomain*> consumer_ids(
          all_consumer_ids.begin(), all_consumer_ids.end());
      for (auto producer : producers) {
        auto all_producer_ids = ir_utils::allIDsOf(producer);
        // Change data structure for IterDomainGraph::buildMapBetween
        VectorOfUniqueEntries<IterDomain*> producer_ids(
            all_producer_ids.begin(), all_producer_ids.end());

        auto p2c = id_graph_.buildMapBetween(
            producer_ids, consumer_ids, IdMappingMode::PERMISSIVE);

        consumers_map_.insert(p2c.begin(), p2c.end());
      }
    }
  }
}

void ComputeAtMap::buildConcreteIds() {
  // For the exact map just select the first ID since they're all exactly the
  // same size, it does not matter which is selected. This should be
  // run-to-run deterministic but which ID gets selected her depends on the
  // traversal order generating the set (compute at map build).
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdSets(IdMappingMode::EXACT).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    concrete_id_cache_[disjoint_set_shared_ptr] = first_id;
  }

  // The following two algorithms seem quite wasteful. Should find a more
  // efficient way to compute concrete IDs.
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdSets(IdMappingMode::PERMISSIVE).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  // Same as exact computation
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdSets(IdMappingMode::ALMOSTEXACT).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::ALMOSTEXACT);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.getDisjointIdSets(IdMappingMode::LOOP).disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
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
  auto disjoint_sets = ca_map.idGraph().getDisjointIdSets(mode).disjointSets();
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

// TODO: Deduplicate with IterDomainGraph::toString()
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

const IdGroup ComputeAtMap::disjointSetOf(IterDomain* id, IdMappingMode mode)
    const {
  auto disjoint_set_pair = id_graph_.getDisjointIdSet(id, mode);
  TORCH_INTERNAL_ASSERT(
      disjoint_set_pair.second,
      id->toString(),
      " has not been processed in this Compute At Map, yet the disjoint set for it was requested in mode: ",
      mode);
  return disjoint_set_pair.first;
}

IdGroups ComputeAtMap::getInputDisjointSetsOf(
    IdGroup of_id,
    bool stop_at_rfactor) {
  IdGroups input_disjoint_sets;

  VectorOfUniqueEntries<IterDomain*> inputs;
  // This deque could be VectorOfUniqueEntries
  std::deque<IdGroup> to_visit({of_id});
  std::unordered_set<IdGroup> visited;
  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.emplace(currently_visiting).second) {
      continue;
    }
    auto defs_pair = id_graph_.getIterDomainGroupDefinitions(
        currently_visiting, IdMappingMode::EXACT);

    // If there's no definition, we've found an input.
    if (!defs_pair.second || defs_pair.first.empty()) {
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
    IdGroups producers_of_currently_visiting;

    for (auto def_group : defs_pair.first) {
      if (def_group->size() == 0) {
        continue;
      }
      auto first_def = def_group->front();
      auto id_inps = ir_utils::filterByType<IterDomain>(first_def->inputs());
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

IdGroups ComputeAtMap::getAllDisjointSetProducers(const IdGroups& exact_sets) {
  // This deque could be VectorOfUniqueEntries
  std::deque<IdGroup> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  IdGroups visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto defs_pair = id_graph_.getIterDomainGroupDefinitions(
        currently_visiting, IdMappingMode::EXACT);

    if (!defs_pair.second) {
      continue;
    }

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    IdGroups producers_of_currently_visiting;

    for (auto def_group : defs_pair.first) {
      if (def_group->size() == 0) {
        continue;
      }
      auto first_def = def_group->front();
      auto id_inps = ir_utils::filterByType<IterDomain>(first_def->inputs());
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

IdGroups ComputeAtMap::getAllDisjointSetConsumers(const IdGroups& exact_sets) {
  // This deque could be VectorOfUniqueEntries
  std::deque<IdGroup> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  IdGroups visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto uses_pair = id_graph_.getIterDomainGroupUses(
        currently_visiting, IdMappingMode::EXACT);

    if (!uses_pair.second) {
      continue;
    }

    // Traverse consumers of current disjoint set and collect unique exact
    // disjoint set consumers
    IdGroups consumers_of_currently_visiting;

    for (auto use_group : uses_pair.first) {
      if (use_group->size() == 0) {
        continue;
      }
      auto first_use = use_group->front();
      auto id_outs = ir_utils::filterByType<IterDomain>(first_use->outputs());
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
          return getDisjointIdSets(IdMappingMode::PERMISSIVE)
              .permissiveAreMapped(id, consumer_id);
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
       id_graph_.getDisjointIdSets(IdMappingMode::LOOP).disjointSets()) {
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
