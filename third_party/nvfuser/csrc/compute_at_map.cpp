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

namespace debug_print {
// A few compressed printing utilities to show critical uniqueness information.
// i.e. being able to tell slight differences between groups we're working with.

template <typename T>
std::string ptrStringShort(const T* ptr) {
  std::stringstream ss;
  ss << ptr;
  return "0x." + ss.str().substr(9);
}

std::string idGroupStringShort(const IdGroup& id_group) {
  std::stringstream ss;
  ss << ptrStringShort(id_group.get()) << "(idg){";
  bool first = true;
  for (auto id : *id_group) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << id->name();
  }
  ss << "}";
  return ss.str();
}

std::string idsStringShort(const VectorOfUniqueEntries<IterDomain*>& id_group) {
  std::stringstream ss;
  ss << "{";
  bool first = true;
  for (auto id : id_group) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << id->name();
  }
  ss << "}";
  return ss.str();
}

std::string idGroupsStringShort(const IdGroups& id_groups) {
  std::stringstream ss;
  ss << ptrStringShort(&id_groups) << "(idgs){";
  bool first = true;
  for (auto id_group : id_groups) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << idGroupStringShort(id_group);
  }
  ss << "}";
  return ss.str();
}

std::string exprGroupStringShort(ExprGroup expr) {
  std::stringstream ss;
  ss << ptrStringShort(expr.get()) << "(exprg){";
  bool first = true;
  for (auto expr_ : *expr) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << expr_->name();
  }

  ss << "}";
  return ss.str();
}

std::string exprGroupStringShort(
    const IterDomainGraphs& id_graph,
    ExprGroup expr_group,
    IdMappingMode mode) {
  std::stringstream ss;
  auto inputs = id_graph.idGraph(mode).inputGroups(expr_group);
  auto outputs = id_graph.idGraph(mode).outputGroups(expr_group);
  ss << idGroupsStringShort(inputs) << " -" << exprGroupStringShort(expr_group)
     << "-> " << idGroupsStringShort(outputs);
  return ss.str();
}

std::string exprGroupsStringShort(
    const IterDomainGraphs& id_graph,
    ExprGroups expr_groups,
    IdMappingMode mode) {
  std::stringstream ss;
  ss << "{\n";
  for (auto expr_group : expr_groups) {
    ss << "  " << exprGroupStringShort(id_graph, expr_group, mode) << "\n";
  }
  ss << "}";
  return ss.str();
}

std::string definitionsToString(
    const IterDomainGraphs& id_graph,
    IdMappingMode mode) {
  std::stringstream ss;
  ss << "All Exprs registered as a definition in mode " << mode << ": "
     << std::endl;
  ExprGroups defs;
  for (auto id_group : id_graph.idGraph(mode).disjointIdSets().disjointSets()) {
    auto definition_pair =
        id_graph.idGraph(mode).iterDomainGroupDefinitions(id_group);
    if (definition_pair.second) {
      for (auto expr_group : definition_pair.first) {
        defs.pushBack(expr_group);
      }
    }
  }
  for (auto expr : defs) {
    ss << exprGroupStringShort(id_graph, expr, mode) << std::endl;
  }
  return ss.str();
}

std::string usesToString(const IterDomainGraphs& id_graph, IdMappingMode mode) {
  std::stringstream ss;
  ss << "All Exprs registered as a use in mode " << mode << ": " << std::endl;

  for (auto id_group : id_graph.idGraph(mode).disjointIdSets().disjointSets()) {
    auto uses_pair = id_graph.idGraph(mode).iterDomainGroupUses(id_group);
    ss << idGroupStringShort(id_group) << std::endl;
    if (uses_pair.second) {
      for (auto expr_group : uses_pair.first) {
        ss << "  " << exprGroupStringShort(id_graph, expr_group, mode)
           << std::endl;
      }
    }
  }
  return ss.str();
}

} // namespace debug_print

IdGraph::IdGraph(const IdGraph& other) {
  disjoint_ids_ = other.disjoint_ids_;
  disjoint_exprs_ = other.disjoint_exprs_;
  id_uses_ = other.id_uses_;
  id_definitions_ = other.id_definitions_;
  view_rfactor_ids_ = other.view_rfactor_ids_;

  for (auto orig_unique_def_pair : other.unique_definitions_) {
    auto orig_id_group = orig_unique_def_pair.first;
    auto orig_expr_groups = orig_unique_def_pair.second;

    auto new_id_group_pair = disjointIdSet(orig_id_group->front());
    TORCH_INTERNAL_ASSERT(new_id_group_pair.second);
    auto new_id_group = new_id_group_pair.first;

    ExprGroups new_expr_groups;
    for (auto orig_expr_group : orig_expr_groups) {
      auto new_expr_group_pair = disjointExprSet(orig_expr_group->front());
      TORCH_INTERNAL_ASSERT(new_expr_group_pair.second);
      new_expr_groups.pushBack(new_expr_group_pair.first);
    }

    unique_definitions_[new_id_group] = new_expr_groups;
  }

  for (auto orig_unique_use_pair : other.unique_uses_) {
    auto orig_id_group = orig_unique_use_pair.first;
    auto orig_expr_groups = orig_unique_use_pair.second;

    auto new_id_group_pair = disjointIdSet(orig_id_group->front());
    TORCH_INTERNAL_ASSERT(new_id_group_pair.second);
    auto new_id_group = new_id_group_pair.first;

    ExprGroups new_expr_groups;
    for (auto orig_expr_group : orig_expr_groups) {
      auto new_expr_group_pair = disjointExprSet(orig_expr_group->front());
      TORCH_INTERNAL_ASSERT(new_expr_group_pair.second);
      new_expr_groups.pushBack(new_expr_group_pair.first);
    }

    unique_uses_[new_id_group] = new_expr_groups;
  }
}

IdGraph& IdGraph::operator=(const IdGraph& other) {
  disjoint_ids_.clear();
  disjoint_exprs_.clear();
  unique_definitions_.clear();
  unique_uses_.clear();
  id_uses_.clear();
  id_definitions_.clear();
  view_rfactor_ids_.clear();
  IdGraph copy(other);
  std::swap(*this, copy);
  return *this;
}

const DisjointSets<IterDomain*>& IdGraph::disjointIdSets() const {
  return disjoint_ids_;
}

DisjointSets<IterDomain*>& IdGraph::disjointIdSets() {
  return disjoint_ids_;
}

std::pair<IdGroup, bool> IdGraph::disjointIdSet(IterDomain* id) const {
  auto disjoint_set_it = disjoint_ids_.disjointSetMap().find(id);
  if (disjoint_set_it == disjoint_ids_.disjointSetMap().end()) {
    return std::make_pair(IdGroup(nullptr), false);
  }
  return std::make_pair(disjoint_set_it->second, true);
}

const DisjointSets<Expr*>& IdGraph::disjointExprSets() const {
  return disjoint_exprs_;
}

DisjointSets<Expr*>& IdGraph::disjointExprSets() {
  return disjoint_exprs_;
}

std::pair<ExprGroup, bool> IdGraph::disjointExprSet(Expr* expr) const {
  auto disjoint_set_it = disjoint_exprs_.disjointSetMap().find(expr);
  if (disjoint_set_it == disjoint_exprs_.disjointSetMap().end()) {
    return std::make_pair(ExprGroup(nullptr), false);
  }
  return std::make_pair(disjoint_set_it->second, true);
}

ExprGroups IdGraph::toGroups(const VectorOfUniqueEntries<Expr*>& exprs) const {
  ExprGroups expr_groups;
  for (auto expr : exprs) {
    auto disjoint_set_pair = disjointExprSet(expr);
    if (disjoint_set_pair.second) {
      expr_groups.pushBack(disjoint_set_pair.first);
    }
  }
  return expr_groups;
}

IdGroups IdGraph::toGroups(
    const VectorOfUniqueEntries<IterDomain*>& ids) const {
  IdGroups id_groups;
  for (auto id : ids) {
    auto disjoint_set_pair = disjointIdSet(id);
    if (disjoint_set_pair.second) {
      id_groups.pushBack(disjoint_set_pair.first);
    }
  }
  return id_groups;
}

IdGroups IdGraph::outputGroups(ExprGroup expr) const {
  VectorOfUniqueEntries<IterDomain*> id_outputs;
  for (auto id_output :
       ir_utils::filterByType<IterDomain>(expr->front()->outputs())) {
    id_outputs.pushBack(id_output);
  }
  return toGroups(id_outputs);
}

IdGroups IdGraph::inputGroups(ExprGroup expr) const {
  VectorOfUniqueEntries<IterDomain*> id_inputs;
  for (auto id_input :
       ir_utils::filterByType<IterDomain>(expr->front()->inputs())) {
    id_inputs.pushBack(id_input);
  }
  return toGroups(id_inputs);
}

ExprGroups IdGraph::allUsesOf(const IdGroups& of) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_uses_pair = iterDomainGroupUses(of_id_group);
    if (group_uses_pair.second) {
      to_visit.pushBack(group_uses_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto output_ids = outputGroups(current_expr);
    for (auto output_id : output_ids) {
      auto group_uses_pair = iterDomainGroupUses(output_id);
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

ExprGroups IdGraph::allDefinitionsOf(const IdGroups& of) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_defs_pair = iterDomainGroupDefinitions(of_id_group);
    if (group_defs_pair.second) {
      to_visit.pushBack(group_defs_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto input_ids = inputGroups(current_expr);
    for (auto input_id : input_ids) {
      auto group_defs_pair = iterDomainGroupDefinitions(input_id);
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

ExprGroups IdGraph::getExprsBetween(const IdGroups& from, const IdGroups& to)
    const {
  auto all_uses_of_from = allUsesOf(from);
  auto all_definitions_of_to = allDefinitionsOf(to);

  // All of the expressions between from and to. Not all will be used as we
  // just want to define each iter domain group once.
  auto all_exprs = all_uses_of_from.intersect(all_definitions_of_to);

  // There could be IterDomains in from or to that are between other from and
  // to nodes. Make sure to clear those out.
  IdGroups terminating_inputs;
  IdGroups terminating_outputs;
  {
    IdGroups not_inputs;
    IdGroups not_outputs;
    IdGroups all_id_groups;

    for (auto expr_group : all_exprs) {
      auto inp_groups = inputGroups(expr_group);
      auto out_groups = outputGroups(expr_group);
      if (inp_groups.intersect(out_groups).size() > 0) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      all_id_groups.pushBack(inp_groups);

      if (inp_groups.empty()) {
        not_outputs.pushBack(inp_groups);
      }

      all_id_groups.pushBack(out_groups);

      if (out_groups.empty()) {
        not_inputs.pushBack(out_groups);
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

  // Return if all output IterDomain groups of an expression group have
  // already been visited
  auto outputsVisited = [&](ExprGroup expr) {
    for (auto id_group : outputGroups(expr)) {
      if (required_ind_exprs_ids.find(id_group) ==
          required_ind_exprs_ids.end()) {
        return false;
      }
    }
    return true;
  };

  auto allIdUsesVisisted = [&](IdGroup id) {
    auto uses_pair = iterDomainGroupUses(id);
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
    for (auto id_group : outputGroups(expr)) {
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
    // Accumulate expressions from all outputs add this expression and set it
    // as current expressions required indexing expressions.
    required_ind_exprs_exprs[expr] = requiredExprsOutputs(expr);
    return true;
  };

  auto processId = [&](IdGroup id) {
    // Track if we've grabed any of the uses required indexing expressions.
    bool initialized = false;
    // Expression group of all indexing expressions required for this iter
    // domain coming back from any of its uses.
    ExprGroups min_groups;

    auto uses_pair = iterDomainGroupUses(id);
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
        auto inp_groups = inputGroups(currently_visiting);
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
        auto definitions_pair = iterDomainGroupDefinitions(currently_visiting);
        if (definitions_pair.second) {
          for (auto def : definitions_pair.first) {
            if (!all_exprs.has(def)) {
              continue;
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
    auto all_uses = iterDomainGroupUses(id);
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
        "Invalid calculation of exprs between, no use found of a provided terminating input: ",
        inp->toString(),
        " expressions cannot be computed.");
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
      auto inputs = inputGroups(currently_visiting);
      if (std::all_of(inputs.begin(), inputs.end(), [&](IdGroup inp_id) {
            return visited.has(inp_id);
          })) {
        something_processed = true;
        sorted_exprs.pushBack(currently_visiting);
        auto outputs = outputGroups(currently_visiting);
        for (auto out_id : outputs) {
          visited.pushBack(out_id);
          auto use_pair = iterDomainGroupUses(out_id);
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

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>> IdGraph::
    buildMapBetween(
        const std::vector<IterDomain*>& from,
        const std::vector<IterDomain*>& to) const {
  std::unordered_map<IterDomain*, IdGroup> from_ids2set;

  for (auto from_id : from) {
    auto from_disjoint_set_pair = disjointIdSet(from_id);
    if (!from_disjoint_set_pair.second) {
      continue;
    }
    from_ids2set[from_id] = from_disjoint_set_pair.first;
  }

  // Map from the sets associated with the IterDomains in to, to those iter
  // domains
  std::unordered_map<IdGroup, VectorOfUniqueEntries<IterDomain*>> set2to_ids;

  for (auto to_id : to) {
    auto to_disjoint_set_pair = disjointIdSet(to_id);
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
  for (auto from_id : from) {
    from_ids2to_ids[from_id] = VectorOfUniqueEntries<IterDomain*>();

    auto from_it = from_ids2set.find(from_id);
    TORCH_INTERNAL_ASSERT(from_it != from_ids2set.end());

    auto from_set = from_it->second;
    auto to_entry_it = set2to_ids.find(from_set);
    if (to_entry_it == set2to_ids.end()) {
      continue;
    }
    from_ids2to_ids[from_id] = to_entry_it->second;
  }
  return from_ids2to_ids;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>> IdGraph::
    buildMapBetween(
        const VectorOfUniqueEntries<IterDomain*>& from,
        const VectorOfUniqueEntries<IterDomain*>& to) const {
  return buildMapBetween(from.vector(), to.vector());
}

std::pair<ExprGroups, bool> IdGraph::iterDomainGroupDefinitions(
    IdGroup id_group) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto definitions_it = unique_definitions_.find(id_group);
  if (definitions_it == unique_definitions_.end()) {
    return null_return;
  }

  return std::make_pair(definitions_it->second, true);
}

std::pair<ExprGroups, bool> IdGraph::iterDomainGroupUses(
    IdGroup id_group) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto uses_it = unique_uses_.find(id_group);
  if (uses_it == unique_uses_.end()) {
    return null_return;
  }

  return std::make_pair(uses_it->second, true);
}

// TODO: Improve and extend to include other information.
std::string IdGraph::toString() const {
  std::stringstream ss;
  ss << "IdGraph { \n";
  ss << "Disjoint Id Set " << disjoint_ids_.toString() << std::endl;
  ss << " } IdGraph\n" << std::endl;
  return ss.str();
}

std::vector<std::vector<IterDomain*>> IdGraph::isTrivialExpr(Expr* expr) {
  std::vector<std::vector<IterDomain*>> mapped_ids;
  if (auto merge = dynamic_cast<Merge*>(expr)) {
    if (merge->inner()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->outer(), merge->out()});
    }
    if (merge->outer()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->inner(), merge->out()});
    }
  } else if (auto split = dynamic_cast<Split*>(expr)) {
    if (split->factor()->isOneInt() && split->startOffset()->isZeroInt() &&
        split->stopOffset()->isZeroInt()) {
      if (split->innerSplit()) {
        mapped_ids.push_back({split->in(), split->outer()});
      } else {
        mapped_ids.push_back({split->in(), split->inner()});
      }
    }
  } else if (auto swizzle = dynamic_cast<Swizzle2D*>(expr)) {
    if (swizzle->swizzleType() == Swizzle2DType::NoSwizzle ||
        swizzle->swizzleMode() == SwizzleMode::NoSwizzle) {
      mapped_ids.push_back({swizzle->inX(), swizzle->outX()});
      mapped_ids.push_back({swizzle->inY(), swizzle->outY()});
    }
  }
  return mapped_ids;
}

// TODO: Add explicit id_definitions_ and id_uses_
void IdGraph::initializeId(
    IterDomain* id,
    const VectorOfUniqueEntries<Expr*>& definitions,
    const VectorOfUniqueEntries<Expr*>& uses) {
  auto id_disjoint_set = disjointIdSets().initializeSet(id).first->second;

  ExprGroups def_groups;
  for (auto def : definitions) {
    auto expr_set = disjointExprSets().initializeSet(def).first->second;
    def_groups.pushBack(expr_set);
  }
  unique_definitions_[id_disjoint_set] = def_groups;

  ExprGroups use_groups;
  for (auto use : uses) {
    auto expr_set = disjointExprSets().initializeSet(use).first->second;
    use_groups.pushBack(expr_set);
  }
  unique_uses_[id_disjoint_set] = use_groups;
}

bool IdGraph::exprsMap(Expr* first, Expr* second, bool forward) const {
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
              return !disjointIdSets().permissiveAreMapped(
                  id_pair.first, id_pair.second);
            })) {
      return false;
    }
  }

  if (first->isA<Merge>() && !forward) {
    // Can't back prop through merge without making sure one input actually
    // matches. This can be done on a map or extent basis.
    auto merge0 = first->as<Merge>();
    auto merge1 = second->as<Merge>();

    auto extent_0o = merge0->outer()->extent();
    auto extent_0i = merge0->inner()->extent();
    auto extent_1o = merge1->outer()->extent();
    auto extent_1i = merge1->inner()->extent();

    auto extent_0_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluateInt() == extent_1o->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->outer(), merge1->outer());

    auto extent_1_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluateInt() == extent_1i->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->inner(), merge1->inner());

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

ExprGroups IdGraph::uniqueDefinitions(IdGroup group) const {
  auto unique_defs_it = unique_definitions_.find(group);
  TORCH_INTERNAL_ASSERT(
      unique_defs_it != unique_definitions_.end(),
      "Definition not found for IdGroup: ",
      group->toString());
  return unique_defs_it->second;
}

ExprGroups IdGraph::uniqueUses(IdGroup group) const {
  auto unique_uses_it = unique_uses_.find(group);
  TORCH_INTERNAL_ASSERT(
      unique_uses_it != unique_definitions_.end(),
      "Uses not found for IdGroup: ",
      group->toString());
  return unique_uses_it->second;
}

void IdGraph::mapExprs(Expr* expr0, Expr* expr1) {
  if (expr0 == expr1) {
    return;
  }

  if (disjointExprSets().strictAreMapped(expr0, expr1)) {
    return;
  }

  // TODO: make these class functions for convenience, there are too many
  // asserts in this file.
  auto assert_get_expr_group = [&](Expr* expr) {
    auto expr_group_pair = disjointExprSet(expr);
    TORCH_INTERNAL_ASSERT(
        expr_group_pair.second, "Could not find entry for expression: ", expr);
    return expr_group_pair.first;
  };

  auto assert_get_id_group = [&](IterDomain* id) {
    auto id_group_pair = disjointIdSet(id);
    TORCH_INTERNAL_ASSERT(
        id_group_pair.second, "Could not find entry for IterDomain: ", id);
    return id_group_pair.first;
  };

  ExprGroup expr0_orig_group = assert_get_expr_group(expr0);
  ExprGroup expr1_orig_group = assert_get_expr_group(expr1);

  disjointExprSets().mapEntries(expr0, expr1);

  auto expr_new_group = assert_get_expr_group(expr0);

  // Update unique uses of producers
  IdGroups producers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      producers.pushBack(assert_get_id_group(input_id));
    }
  }

  for (auto producer_group : producers) {
    uniqueUses().at(producer_group).erase(expr0_orig_group);
    uniqueUses().at(producer_group).erase(expr1_orig_group);
    uniqueUses().at(producer_group).pushBack(expr_new_group);
  }

  // Update unique definitinos of consumers
  IdGroups consumers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      consumers.pushBack(assert_get_id_group(output_id));
    }
  }

  for (auto consumer_group : consumers) {
    uniqueDefinitions().at(consumer_group).erase(expr0_orig_group);
    uniqueDefinitions().at(consumer_group).erase(expr1_orig_group);
    uniqueDefinitions().at(consumer_group).pushBack(expr_new_group);
  }
}

void IdGraph::mapIds(IterDomain* id0, IterDomain* id1) {
  if (id0 == id1) {
    return;
  }

  if (disjointIdSets().strictAreMapped(id0, id1)) {
    return;
  }
  // Definitions and uses are based on the groups of id0 and id1, don't merge
  // them into a single group until we grab all definitions and uses for later
  // processing.
  auto orig_id_group0 = disjointIdSet(id0).first;
  auto orig_id_group1 = disjointIdSet(id1).first;
  ExprGroups orig_defs0 = uniqueDefinitions(orig_id_group0);
  ExprGroups orig_defs1 = uniqueDefinitions(orig_id_group1);
  ExprGroups orig_uses0 = uniqueUses(orig_id_group0);
  ExprGroups orig_uses1 = uniqueUses(orig_id_group1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointIdSets().mapEntries(id0, id1);
  auto new_id_group = disjointIdSet(id0).first;

  unique_definitions_.erase(orig_id_group0);
  unique_definitions_.erase(orig_id_group1);
  unique_uses_.erase(orig_id_group0);
  unique_uses_.erase(orig_id_group1);

  unique_definitions_[new_id_group] = orig_defs0.computeUnion(orig_defs1);
  unique_uses_[new_id_group] = orig_uses0.computeUnion(orig_uses1);

  // Propagate on uses
  if (orig_uses0.size() > 0 || orig_uses1.size() > 0) {
    if (orig_uses0.size() > 0 && orig_uses1.size() > 0) {
      for (auto use_group_1 : orig_uses1) {
        if (orig_uses0.has(use_group_1)) {
          continue;
        }

        for (auto use_group_0 : orig_uses0) {
          auto use0 = use_group_0->front();
          auto use1 = use_group_1->front();
          if (exprsMap(use0, use1, true)) {
            mapExprs(use0, use1);
            mapThroughExpr(use0, use1, true);
          }
        }
      }
    }
  }

  // Propagate on definitions
  if (orig_defs0.size() > 0 || orig_defs1.size() > 0) {
    if (orig_defs0.size() > 0 && orig_defs1.size() > 0) {
      for (auto def_group_1 : orig_defs1) {
        if (orig_defs0.has(def_group_1)) {
          continue;
        }

        for (auto def_group_0 : orig_defs0) {
          auto def0 = def_group_0->front();
          auto def1 = def_group_1->front();
          if (exprsMap(def0, def1, false)) {
            mapExprs(def0, def1);
            mapThroughExpr(def0, def1, false);
          }
        }
      }
    }
  }
}

bool IdGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward)) {
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
    mapIds(first_ids[out_i], second_ids[out_i]);
  }

  return true;
}

void IterDomainGraphs::assertNoSelfMapping() {
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

void IdGraph::mapThroughLoopSwizzles() {
  for (auto use_pairs : unique_uses_) {
    auto use_groups = use_pairs.second;
    for (auto use_group : use_groups) {
      for (auto use : *use_group) {
        if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(use)) {
          // Map each input to its corresponding output on the given
          // disjoint set if this is a loop swizzle. Loop swizzles don't impact
          // indexing, only iteration order.
          if (swizzle_2d->swizzleMode() == SwizzleMode::Loop) {
            mapIds(swizzle_2d->inX(), swizzle_2d->outX());
            mapIds(swizzle_2d->inY(), swizzle_2d->outY());
          }
        }
      }
    }
  }
}

IterDomainGraphs::IterDomainGraphs(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool allow_self_mapping) {
  build(exprs, additional_tvs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

IterDomainGraphs::IterDomainGraphs(
    const std::vector<Expr*>& exprs,
    bool allow_self_mapping)
    : IterDomainGraphs(exprs, {}, allow_self_mapping) {}

IterDomainGraphs::IterDomainGraphs(Fusion* fusion, bool allow_self_mapping) {
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

const IdGraph& IterDomainGraphs::idGraph(IdMappingMode mode) const {
  auto graph_it = id_graphs_.find(mode);
  TORCH_INTERNAL_ASSERT(graph_it != id_graphs_.end());
  return graph_it->second;
}

IdGraph& IterDomainGraphs::idGraph(IdMappingMode mode) {
  auto graph_it = id_graphs_.find(mode);
  TORCH_INTERNAL_ASSERT(graph_it != id_graphs_.end());
  return graph_it->second;
}

Expr* IterDomainGraphs::idUse(IterDomain* id) const {
  auto use_it = id_uses_.find(id);
  if (use_it == id_uses_.end()) {
    return nullptr;
  }
  return use_it->second.front();
}

Expr* IterDomainGraphs::idDef(IterDomain* id) const {
  auto def_it = id_definitions_.find(id);
  if (def_it == id_definitions_.end()) {
    return nullptr;
  }
  return def_it->second.front();
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
// The elements in tv1 {3, 1, 4, 2}, map respectively to the elements in tv2
// {1, 2, 3, 4}. The reason this is so important is it means that generating
// tv3 is no longer a trivially parallelizable problem (if we include the dag
// all the way to tv0). So tv0's axes cannot be inlined across both the tv0
// and tv1 path. This breaks some assumptions we have today in schedulers that
// will assume tv2 can be trivially inlined/parallelized. Instead we'd need to
// take into consideration the effective communication going on here, so that
// we pull multiple values of tv0 to compute tv3.
c10::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraphs& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (id_graph.idGraph(mode).disjointIdSets().permissiveAreMapped(
              id1, id2)) {
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
    const IterDomainGraphs& id_graph) {
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

void IterDomainGraphs::buildIterDomainDefinitionsAndUses(
    const std::vector<TensorView*>& all_tvs) {
  for (auto tv : all_tvs) {
    VectorOfUniqueEntries<IterDomain*> root_domain_ids{
        tv->getRootDomain().begin(), tv->getRootDomain().end()};

    auto all_ids = ir_utils::allIDsOf(tv);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration
        // domain is marked as an rfactor product and is in the rfactor
        // domain, it's a view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->getMaybeRFactorDomain();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          view_rfactor_ids_.emplace(id);
        }
      }

      if (id_definitions_.find(id) == id_definitions_.end()) {
        id_definitions_[id] = {};
      }

      if (id_uses_.find(id) == id_uses_.end()) {
        id_uses_[id] = {};
      }

      auto def = id->definition();

      if (def == nullptr || root_domain_ids.has(id)) {
        continue;
      }

      if (id_definitions_.find(id) == id_definitions_.end()) {
        id_definitions_[id] = {};
      }
      id_definitions_.at(id).pushBack(def);

      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        if (id_uses_.find(inp_id) == id_uses_.end()) {
          id_uses_[inp_id] = {};
        }
        id_uses_.at(inp_id).pushBack(def);
      }
    }
  }
}

// TODO: Extend to include other information.
std::string IterDomainGraphs::toString() const {
  std::stringstream ss;
  ss << "IterDomainGraphs { \n";
  // for (auto set : disjoint_ids_) {
  //   ss << "Set " << set.first << ": " << std::endl;
  //   ss << set.second.toString() << std::endl;
  // }
  ss << " } IterDomainGraphs\n" << std::endl;
  return ss.str();
}

// Replay Expr but with the inputs provided.
Expr* IterDomainGraphs::addReplayAs(
    const std::vector<IterDomain*>& new_inputs,
    Expr* expr) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  auto orig_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
  std::vector<IterDomain*> orig_input_ids(
      orig_inputs.begin(), orig_inputs.end());

  {
    TORCH_INTERNAL_ASSERT(
        new_inputs.size() == orig_input_ids.size(),
        "Invalid number of inputs: ",
        new_inputs.size(),
        " does not match number of iter domain inputs for ",
        expr->toString());

    VectorOfUniqueEntries<IterDomain*> all_inputs{
        orig_input_ids.begin(), orig_input_ids.end()};

    all_inputs.pushBack(VectorOfUniqueEntries<IterDomain*>{
        new_inputs.begin(), new_inputs.end()});

    for (auto mode : initialized_modes) {
      for (auto inp : all_inputs) {
        TORCH_INTERNAL_ASSERT(
            idGraph(mode).disjointIdSet(inp).second,
            "All inputs for replay need to be initialized in all graphs, ",
            inp->toString(),
            " was not found in mode: ",
            mode);
      }
    }
  }

  // Create the new expression with provided inputs
  auto replay = ReplayTransform::replayAs(new_inputs, expr);

  for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
    id_definitions_[out_id] = {replay};
    id_uses_[out_id] = {};
  }

  // Add the expression to the uses of the inputs
  for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
    id_uses_.at(inp_id).pushBack(replay);
  }

  // Initialize output iter domains in the graphs
  for (auto mode : initialized_modes) {
    idGraph(mode).disjointExprSets().initializeSet(replay);
    auto replay_group = idGraph(mode).disjointExprSet(replay).first;

    // Initialize output ids in map
    for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      idGraph(mode).initializeId(out_id, {replay}, {});
    }

    // Update uses of the inputs in the graphs
    for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      auto inp_group = idGraph(mode).disjointIdSet(inp_id).first;
      idGraph(mode).uniqueUses().at(inp_group).pushBack(replay_group);
    }

    // Propagate through all the uses of the iter domain groups of the inputs
    // with the new expression.
    auto& graph = idGraph(mode);
    // Gather all use expressions from inputs
    VectorOfUniqueEntries<Expr*> representative_uses;
    for (auto inp : new_inputs) {
      auto uses_pair =
          graph.iterDomainGroupUses(graph.disjointIdSet(inp).first);
      if (uses_pair.second) {
        for (auto use_group : uses_pair.first) {
          representative_uses.pushBack(use_group->front());
        }
      }
    }

    for (auto expr : representative_uses) {
      if (graph.exprsMap(expr, replay, true)) {
        graph.mapExprs(expr, replay);
        graph.mapThroughExpr(expr, replay, true);
      }
    }
  }

  return replay;
}

IdGraph IterDomainGraphs::initializeIdGraph() {
  IdGraph id_graph;

  for (auto definition_entry : id_definitions_) {
    auto id = definition_entry.first;
    auto defs = definition_entry.second;
    auto uses_it = id_uses_.find(id);
    TORCH_INTERNAL_ASSERT(
        uses_it != id_uses_.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeId(id, defs, uses_it->second);
  }

  return id_graph;
}

void IterDomainGraphs::buildExactMap(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    // Map siblings, as all other tv output domains must match the first tv
    // outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    for (auto other_tv_output : other_tv_outputs) {
      // Sibling tv's must be exactly mapped with eachother so simply zip
      // their leaf iter domains.

      TORCH_INTERNAL_ASSERT(
          other_tv_output->getRootDomain().size() ==
              c_tv->getRootDomain().size(),
          "Multiple outputs with mismatched TV domains is not supported.");

      for (auto domain_i : c10::irange(c_tv->getRootDomain().size())) {
        auto c_id = c_tv->getRootDomain()[domain_i];
        auto o_id = other_tv_output->getRootDomain()[domain_i];
        idGraph(IdMappingMode::EXACT).mapIds(o_id, c_id);
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
        idGraph(IdMappingMode::EXACT).mapIds(c_id, p_id);
      }
    }

    idGraph(IdMappingMode::EXACT).mapThroughLoopSwizzles();
  }
}

void IterDomainGraphs::buildPermissiveMap(const std::vector<Expr*>& exprs) {
  idGraph(IdMappingMode::PERMISSIVE) = idGraph(IdMappingMode::ALMOSTEXACT);

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
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }

      // TODO: Should this just get rolled up in the forwarding map now?
      for (auto entry : permissive_forwarding.producer_compliment_map) {
        for (auto entry_2 : entry.second) {
          idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry_2);
        }
      }

      for (auto entry : permissive_forwarding.consumer_forwarding_map) {
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }

      // TODO: Should this just get rolled up in the forwarding map now?
      for (auto entry : permissive_forwarding.consumer_compliment_map) {
        for (auto entry_2 : entry.second) {
          idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry_2);
        }
      }

      auto permissive_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv);

      for (auto entry : permissive_c2p_root_map.mapConsumerToProducer(
               c_tv->domain(), p_tv->domain())) {
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }
    }
  }
  idGraph(IdMappingMode::PERMISSIVE).mapThroughLoopSwizzles();
}

void IterDomainGraphs::buildAlmostExactMap() {
  // Build almost exact map by forwarding through broadcast axes
  idGraph(IdMappingMode::ALMOSTEXACT) = idGraph(IdMappingMode::EXACT);

  VectorOfUniqueEntries<Expr*> exprs;
  for (auto expr :
       idGraph(IdMappingMode::ALMOSTEXACT).disjointExprSets().disjointSets()) {
    exprs.pushBack(expr->front());
  }
  ExprGroups trivial_expr_groups;

  // Map through trivial expressions
  for (auto expr : exprs) {
    auto mapped_ids = IdGraph::isTrivialExpr(expr);
    for (auto mapped_id_group : mapped_ids) {
      for (auto id : mapped_id_group) {
        trivial_expr_groups.pushBack(
            idGraph(IdMappingMode::ALMOSTEXACT).disjointExprSet(expr).first);
        idGraph(IdMappingMode::ALMOSTEXACT).mapIds(mapped_id_group.front(), id);
      }
    }
  }

  // TODO: Clear out expressions that map inputs and outputs to the same group
  // from definitions and uses. They shouldn't be important in traversal.
  // Similar to what's drafted in buildIndexMap
}

void IterDomainGraphs::validateAndPropagatePType() const {
  for (const auto& loop_disjoint_set :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
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

void IterDomainGraphs::build(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs) {
  // Initialize the required sets as if a permissive relationship is never
  // found, then querying an empty permissive map will fail later.
  // Initialize disjoint sets
  for (auto mode : kIdMappingModes) {
    id_graphs_[mode] = IdGraph();
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
  FusionGuard::getCurFusion()->print();
  // Add uses and definitions to all iter domains.
  buildIterDomainDefinitionsAndUses(all_tvs);

  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  idGraph(IdMappingMode::EXACT) = initializeIdGraph();

  buildExactMap(tv_exprs);

  buildAlmostExactMap();

  buildPermissiveMap(tv_exprs);

  // Only build loop map during lowering
  if (FusionGuard::getCurFusion()->isA<kir::Kernel>()) {
    idGraph(IdMappingMode::LOOP) = initializeIdGraph();

    // Find loops that need to be promoted because of broadcast resolution,
    // figure out what that resolution should look like, compute IDs for it if
    // necessary.
    buildLoopPromotionMap(tv_exprs);

    std::cout << "Built Loop map:" << std::endl;
    for (auto entry :
         idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
      std::cout << entry->toString() << std::endl;
      std::cout << "-> " << loop_promotion_map_.at(entry) << std::endl;
    }

    TORCH_INTERNAL_ASSERT(false);

    validateAndPropagatePType();
  }

  // Debug, make sure there's no self mapping in TensorView's during lowering
  // that would invalidate lowering assumptions.
  self_mapping_info_ = findFirstSelfMapping(all_tvs, *this);
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

std::unordered_map<IdGroup, IdGroups> IterDomainGraphs::
    buildCoveredAlmostExact() {
  // Helper functions.
  auto producerIdGroups = [&](IdGroup id_group) {
    IdGroups producer_groups;
    auto definition_pair_it = idGraph(IdMappingMode::ALMOSTEXACT)
                                  .iterDomainGroupDefinitions(id_group);
    if (!definition_pair_it.second) {
      return producer_groups;
    }
    for (auto def_group : definition_pair_it.first) {
      auto inp_groups =
          idGraph(IdMappingMode::ALMOSTEXACT).inputGroups(def_group);
      producer_groups.pushBack(inp_groups);
    }
    return producer_groups;
  };

  auto consumerIdGroups = [&](IdGroup id_group) {
    IdGroups consumer_groups;
    auto uses_pair_it =
        idGraph(IdMappingMode::ALMOSTEXACT).iterDomainGroupUses(id_group);
    if (!uses_pair_it.second) {
      return consumer_groups;
    }
    for (auto use_group : uses_pair_it.first) {
      auto out_groups =
          idGraph(IdMappingMode::ALMOSTEXACT).outputGroups(use_group);
      consumer_groups.pushBack(out_groups);
    }
    return consumer_groups;
  };

  // Start at terminating inputs of the almost exact graph and almost exact
  // entries that are rfactor nodes. Propagate and accumulate these nodes
  // through consumers.
  //
  // The almost exact entries covered by an iteration domain is effectively
  // all the iteration domains this domain relies on. Initialize broadcast
  // entries to not cover any domains.
  std::unordered_map<IdGroup, IdGroups> covered_almost_exact_entries;

  // We will traverse over the almost exact set expressions. Save where we
  // want to start traversal:
  IdGroups to_visit;
  // Initialize covered groups
  for (auto almost_exact_set :
       idGraph(IdMappingMode::ALMOSTEXACT).disjointIdSets().disjointSets()) {
    // what broadcast domains cover doesn't matter
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
    auto def_pair = idGraph(IdMappingMode::ALMOSTEXACT)
                        .iterDomainGroupDefinitions(almost_exact_set);
    if (!def_pair.second) {
      covered_almost_exact_entries[almost_exact_set] = {almost_exact_set};
      to_visit.pushBack(consumerIdGroups(almost_exact_set));
      continue;
    }

    for (auto def : def_pair.first) {
      // If all definitions are self mapping (can happen with
      // merging our splitting with a broadcast/ dim of size 1)
      // then this group is an input.
      auto inp_groups = idGraph(IdMappingMode::ALMOSTEXACT).inputGroups(def);
      if (std::find(inp_groups.begin(), inp_groups.end(), almost_exact_set) ==
          inp_groups.end()) {
        goto loop_continue;
      }
    }

    covered_almost_exact_entries[almost_exact_set] = {almost_exact_set};
    to_visit.pushBack(consumerIdGroups(almost_exact_set));

  loop_continue:;
  }

  // Starting from the initialized inputs propagate forward from those inputs to
  // mark what every iter domain in the graph covers. This will be used in later
  // analysis.
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
  return covered_almost_exact_entries;
}

void IterDomainGraphs::buildLoopPromotionMap(const std::vector<Expr*>& exprs) {
  // == Stage 1 ==: This stage is primarily like concrete ID finding. We're
  // going to initialize all the terminating inputs and all of the rfactor
  // groups in the almost exact map to simply "cover" themselves. Cover really
  // just means "inputs" to those iter domains. We're trying to find loop maps
  // that cover all the concrete IDs that they should loop over in part or
  // entirely.
  auto covered_almost_exact_entries = buildCoveredAlmostExact();

  // == Stage 2 ==: Calculate which iter domains are shared across producers
  // and consumers. Shared iter domains are from inlining, they're the iter
  // domains within the compute at position and max produce at position of
  // tensor views and all the iter domains required to generate those iter
  // domains. (p2c_ca_permissive_maps)
  //
  // We need to figure out within all of those which ones are undergoing a
  // broadcast resolution process. These are the domains that are tricky to
  // resolve as producer leaf nodes need to be promoted to include that
  // resolved broadcast when they're inlined into their consumers resulting in
  // being inlined into that resolved broadcast..

  // Track which root iter domains are resolved and inlined. Track what
  // they're resolved to.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_root_broadcast_resolution_map;

  // Track all of the p2c mappings through the fusion within those inlined
  // domains.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_ca_permissive_maps;

  // Want to traverse the iter domains when we do promotion in topological
  // order, so we will save that ordering as we populate the above maps.
  VectorOfUniqueEntries<IterDomain*> ordered_p_ca_ids;

  // Utility functions: If provided map already has an entry for provided key,
  // accumulate into that entry the new provided value. Otherwise initialize a
  // new key-value pair in the map.
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

  auto accumulateInMapVec =
      [](std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
             map,
         IterDomain* key,
         const VectorOfUniqueEntries<IterDomain*>& new_values) {
        auto entry_it = map.find(key);
        if (map.find(key) == map.end()) {
          map[key] = new_values;
        } else {
          auto& value = entry_it->second;
          value.pushBack(new_values);
        }
      };

  for (auto expr : exprs) {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
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

        all_producer_ca_deps.insert(
            ca_deps_filter.begin(), ca_deps_filter.end());
      }

      ordered_p_ca_ids.pushBack(all_producer_ca_deps);

      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto resolved_bcast_map = resolvedRootBroadcasts(producer, consumer);
        for (auto entry : resolved_bcast_map) {
          accumulateInMap(
              p2c_root_broadcast_resolution_map, entry.first, entry.second);
          for (auto other_exact_bcast : *idGraph(IdMappingMode::EXACT)
                                             .disjointIdSet(entry.first)
                                             .first) {
            if (all_producer_ca_deps.has(other_exact_bcast)) {
              accumulateInMap(
                  p2c_root_broadcast_resolution_map,
                  other_exact_bcast,
                  entry.second);
            }
          }
        }

        auto p2c_ca_permissive_map = idGraph(IdMappingMode::PERMISSIVE)
                                         .buildMapBetween(
                                             all_producer_ca_deps.vector(),
                                             ir_utils::allIDsOf(consumer));

        for (auto entry : p2c_ca_permissive_map) {
          if (entry.second.size() == 0) {
            continue;
          }
          accumulateInMapVec(p2c_ca_permissive_maps, entry.first, entry.second);
        }
      }
    }
  }

  // == Stage 3 ==: Start accumulating the loop map. Loop map is all about
  // iter domain promotion so we can initialize it easily with the c2p
  // permissive map from processing all the inlined iter domains.
  for (auto entry : p2c_ca_permissive_maps) {
    auto first = entry.first;
    for (auto second : entry.second) {
      idGraph(IdMappingMode::LOOP).mapIds(first, second);
    }
  }

  // == Stage 4 ==: We now need to (potentially) generate the iter domains in
  // the loop map that cover all the almost exact sets that are needed based
  // on broadcast resolution.
  //
  // This analysis is working with three types of disjoint sets now, need to
  // be careful how they're mixed.
  //
  // Loop groups are defined based on groups that share the iter domain
  //   promotion map entries. They should all be promoted to the same type.
  //   They are permissive mapped by definition, but not necessarily almost or
  //   exact mapped.
  //
  // AlmostExact mapping is used to see what iter domains need to be covered by
  //   the replay to cover a full promotion set. We don't need to cover every
  //   exact set in the history, but definitely need to cover all almost exact
  //   sets.
  //
  // Exact mapping is used to perform the actual replay required to cover a full
  //   promotion set. If we have something like (7 * 1) and (1 * 13) the
  //   almost exact map might view these as 7 and 13 without the broadcast
  //   merge. We need the broadcast merge because we need to replay one of
  //   those.

  // Loop map will get updated as we go, make a copy to iterate on and use as a
  // promotion map
  DisjointSets<IterDomain*> loop_map_copy =
      idGraph(IdMappingMode::LOOP).disjointIdSets();
  IdGroups ordered_loop_groups;

  auto disjoint_group_loop_copy = [&loop_map_copy](IterDomain* id) {
    auto disjoint_set_it = loop_map_copy.disjointSetMap().find(id);
    TORCH_INTERNAL_ASSERT(
        disjoint_set_it != loop_map_copy.disjointSetMap().end(),
        id->toString(),
        " not found in promotion map.");
    return disjoint_set_it->second;
  };

  // Order the loop groups we iterate over following producer->consumer
  // root->leaf ordering.
  for (auto id : ordered_p_ca_ids) {
    ordered_loop_groups.pushBack(disjoint_group_loop_copy(id));
  }

  // Promotion map keys are the loop sets in the loop map copy. These sets share
  // share a promoted id.
  std::unordered_map<IdGroup, IterDomain*> promotion_map;

  for (auto orig_loop_group : ordered_loop_groups) {
    // ALMOSTEXACT: All the almost exact sets this group needs to cover
    IdGroups to_cover;

    // EXACT: These are the iter domains in the group furthest in consumer edges
    // when considering producer-consumer connections. (Found by simply
    // propagating the p2c_ca_permissive_maps)
    IdGroups terminal_ids;

    // Group already promoted, no need to continue.
    if (promotion_map.find(orig_loop_group) != promotion_map.end()) {
      continue;
    }

    // Populate terminal_ids and to_cover
    for (auto entry : *orig_loop_group) {
      if (p2c_ca_permissive_maps.find(entry) == p2c_ca_permissive_maps.end()) {
        // Careful, mixing modes in this analysis. EXACT is required to
        // reproduce transformations for this resolution. However, we simply use
        // almost exact map to figure out what IterDomain sets need to be
        // covered.
        auto exact_group_pair =
            idGraph(IdMappingMode::EXACT).disjointIdSet(entry);
        TORCH_INTERNAL_ASSERT(exact_group_pair.second);
        terminal_ids.pushBack(exact_group_pair.first);
        auto almost_exact_group_pair =
            idGraph(IdMappingMode::ALMOSTEXACT).disjointIdSet(entry);
        TORCH_INTERNAL_ASSERT(almost_exact_group_pair.second);
        to_cover.pushBack(
            covered_almost_exact_entries.at(almost_exact_group_pair.first));
      }
    }

    // If there's only one terminal id that has to be the "promoted" id.
    if (terminal_ids.size() == 1) {
      auto promoted_id = terminal_ids.front()->front();
      promotion_map[orig_loop_group] = promoted_id;
      continue;
    }

    // Mark if the promoted id was found and populated in the map so we can
    // stop analysis early.
    bool promotion_found = false;

    for (auto terminal_id : terminal_ids) {
      // Almost exact should be a super set of exact which is where the
      // terminal_id is placed
      auto almost_exact_terminal_pair =
          idGraph(IdMappingMode::ALMOSTEXACT)
              .disjointIdSet(terminal_id->front());
      TORCH_INTERNAL_ASSERT(almost_exact_terminal_pair.second);
      if (to_cover
              .subtract(covered_almost_exact_entries.at(
                  almost_exact_terminal_pair.first))
              .empty()) {
        promotion_map[orig_loop_group] = terminal_id->front();
        promotion_found = true;
        break;
      }
    }

    if (promotion_found) {
      continue;
    }

    // Check if we can more easily build

    // None of the terminal_ids have all the required IterDomains covered.
    // Generate a new IterDomain that satisfies the requirement of covering
    // all of the almost exact sets in "to_cover".

    // Compute all inputs we need to use to replay the terminal ids, start at
    // terminal ids and propagate backwards. Stop at iter domains that don't
    // require promotion, or those already promoted.

    // Grab the iter domains to start the generation from. Do this on the
    // exact map as broadcasts need to be explicitly promoted on replay.
    IdGroups start_point;
    for (auto group : to_cover) {
      for (auto id : *group) {
        start_point.pushBack(
            idGraph(IdMappingMode::EXACT).disjointIdSet(id).first);
      }
    }

    // Check the broadcast promotion map, if to must be covered, then we may
    // have broadcast dimensions we need to promote when we replay. Collect
    // those broadcasts and what they should be promoted to.
    std::unordered_map<IdGroup, IdGroup> bcast_promotion_map;
    for (auto entry : p2c_root_broadcast_resolution_map) {
      auto from = entry.first;
      auto tos = entry.second;
      for (auto to : tos) {
        if (to_cover.has(
                idGraph(IdMappingMode::ALMOSTEXACT).disjointIdSet(to).first)) {
          // TODO: Make sure we're not trying to broadcast the same thing to
          // two different extents.
          bcast_promotion_map
              [idGraph(IdMappingMode::EXACT).disjointIdSet(from).first] =
                  idGraph(IdMappingMode::EXACT).disjointIdSet(to).first;
        }
      }
    }

    for (auto bcast_promo : bcast_promotion_map) {
      start_point.pushBack(bcast_promo.first);
    }

    // Grab all expresions that need to be replayed.
    auto transform_exprs = idGraph(IdMappingMode::EXACT)
                               .getExprsBetween(start_point, terminal_ids);

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
    // the computations. Since 5 and 3 are also split off, full replays will
    // be performed for them too.
    //
    // Finding what we can reuse is a bit challenging. We should be able to
    // reuse iter domains that are promoted, and not replay all the way back
    // from inputs. However, I'm not sure if finding where we can start
    // traversal from is easy. We have a local_promotion_map that is not the
    // global_promotion_map. I don't believe these are the same in all cases.
    //
    // Leaving the bad complexity here for now, but should revisit and fix as
    // this could blow up quickly.
    std::unordered_map<IdGroup, IdGroup> local_promotion_map;

    // Perform replay
    for (auto transform_expr : transform_exprs) {
      std::vector<IterDomain*> new_input_ids;
      for (auto inp_group :
           idGraph(IdMappingMode::EXACT).inputGroups(transform_expr)) {
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

      auto replayed_expr = addReplayAs(new_input_ids, transform_expr->front());

      auto orig_outputs_ids =
          ir_utils::filterByType<IterDomain>(transform_expr->front()->outputs())
              .vector();

      auto new_outputs_ids =
          ir_utils::filterByType<IterDomain>(replayed_expr->outputs()).vector();

      TORCH_INTERNAL_ASSERT(orig_outputs_ids.size() == new_outputs_ids.size());

      // Add outputs to promotion map
      for (auto id_i : c10::irange(orig_outputs_ids.size())) {
        auto orig_set_pair =
            idGraph(IdMappingMode::EXACT).disjointIdSet(orig_outputs_ids[id_i]);
        auto replay_set_pair =
            idGraph(IdMappingMode::EXACT).disjointIdSet(new_outputs_ids[id_i]);
        TORCH_INTERNAL_ASSERT(orig_set_pair.second && replay_set_pair.second);
        local_promotion_map[orig_set_pair.first] = replay_set_pair.first;
      }
    }

    for (auto terminal_id : terminal_ids) {
      // TODO: Uncertain if this check is sufficient. In the case that there's
      // multiple terminal id's, could they cover different domains?
      if (local_promotion_map.find(terminal_id) != local_promotion_map.end()) {
        promotion_map[orig_loop_group] =
            local_promotion_map.at(terminal_id)->front();
        promotion_found = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        promotion_found,
        "Error computing promoted iter domain for group: ",
        orig_loop_group->toString());
  }

  // == Stage 5 ==: At this point all the inlined loops have been promoted.
  // However producer's may have transformations that are on top of now
  // promoted iter domains. Replay those transformations on top of the
  // promoted ids and potentially continue the promoted map to extend outside
  // the directly inlined loops.

  // Convert promotion map to be on an IterDomain by IterDomain basis to make
  // it easier to directly replay tensor views.
  std::unordered_map<IterDomain*, IterDomain*> id_promotion_map;

  for (auto promotion_map_entry : promotion_map) {
    for (auto from_id : *promotion_map_entry.first) {
      auto to_id = promotion_map_entry.second;
      if (!idGraph(IdMappingMode::ALMOSTEXACT)
               .disjointIdSets()
               .permissiveAreMapped(from_id, to_id)) {
        id_promotion_map[from_id] = to_id;
      }
    }
  }

  for (auto expr : exprs) {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // We don't just care about the inlined axes in the tensor view but all
      // axes that are shared with other tensor views, so go to the higher of
      // compute at and max produce at.
      auto shared_loop_pos = std::max(
          producer->getMaxProducerPosition(), producer->getComputeAtPosition());
      if (producer->nDims() == shared_loop_pos || shared_loop_pos == 0) {
        // No leaf promotions needed, don't process
        continue;
      }

      auto domain = producer->domain()->domain();
      auto root = producer->getMaybeRFactorDomain();

      // Grab all iter domains that might already be promoted
      VectorOfUniqueEntries<IterDomain*> all_producer_ca_deps;
      {
        auto ca_dep_vals = DependencyCheck::getAllValsBetween(
            {root.begin(), root.end()},
            {domain.begin(), domain.begin() + shared_loop_pos});

        auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);

        all_producer_ca_deps.insert(
            ca_deps_filter.begin(), ca_deps_filter.end());
      }

      // Track all iter domains that actually have a promotion.
      VectorOfUniqueEntries<IterDomain*> all_promoted_ca_deps;

      for (auto id : all_producer_ca_deps) {
        auto promoted_entry_it = id_promotion_map.find(id);
        if (promoted_entry_it == id_promotion_map.end()) {
          continue;
        }

        auto promoted_id = promoted_entry_it->second;
        // If the promoted IterDomain is the same size as this one, no need to
        // promote it.
        if (idGraph(IdMappingMode::ALMOSTEXACT)
                .disjointIdSets()
                .permissiveAreMapped(promoted_id, id)) {
          continue;
        }

        all_promoted_ca_deps.pushBack(id);
        id_promotion_map[id] = promoted_id;
      }

      // Grab all expressions between promoted IterDomains and the iter domains
      // of this tensorview that do not participate in inlining.
      auto transform_exprs = StmtSort::getExprsBetween(
          FusionGuard::getCurFusion(),
          {all_promoted_ca_deps.begin(), all_promoted_ca_deps.end()},
          {domain.begin() + producer->getComputeAtPosition(),
           domain.begin() + producer->nDims()});

      // Perform replay
      for (auto transform_expr : transform_exprs) {
        auto id_inputs =
            ir_utils::filterByType<IterDomain>(transform_expr->inputs());
        IdGroups input_promo_groups;
        for (auto inp : id_inputs) {
          auto loop_set_pair = idGraph(IdMappingMode::LOOP).disjointIdSet(inp);
          if (loop_set_pair.second) {
            input_promo_groups.pushBack(loop_set_pair.first);
          }
        }

        auto id_outputs =
            ir_utils::filterByType<IterDomain>(transform_expr->outputs());
        IdGroups output_promo_groups;
        for (auto out : id_outputs) {
          auto loop_set_pair = idGraph(IdMappingMode::LOOP).disjointIdSet(out);
          if (loop_set_pair.second) {
            output_promo_groups.pushBack(loop_set_pair.first);
          }
        }

        // Due to permissive mapping we could have an input and output of an
        // expression promoted to the same thing. If we re-promote the input
        // then we'll get another incorrect replay. e.g. T2[z], T3[y*z] T2's z,
        // T3's z and T3's y*z will all be in the same promotion group. If we
        // end up replaying T3 we would promote T3's z to y*z, then replay y*z
        // with that promotion resulting in y*y*z
        if (input_promo_groups.intersect(output_promo_groups).size() > 0) {
          continue;
        }

        bool input_promoted = false;
        std::vector<IterDomain*> input_copy{id_inputs.begin(), id_inputs.end()};

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

        auto replay = addReplayAs(input_copy, transform_expr);

        auto orig_outputs_ids =
            ir_utils::filterByType<IterDomain>(transform_expr->outputs())
                .vector();

        auto new_outputs_ids =
            ir_utils::filterByType<IterDomain>(replay->outputs()).vector();

        TORCH_INTERNAL_ASSERT(
            orig_outputs_ids.size() == new_outputs_ids.size());

        // Add outputs to promotion map
        for (auto id_i : c10::irange(orig_outputs_ids.size())) {
          id_promotion_map[orig_outputs_ids[id_i]] = new_outputs_ids[id_i];
        }
      }
    }
  }

  // // == Stage 6 ==: Promotion map is now on an iter domain by iter domain
  // basis. However we need to recolapse this on a loop group basis. Loop
  // groups need to be disjoint based on what loops are actually shared. So a
  // promoted id if generated, cannot be used more than once. Clone the
  // promoted id if it needs to be used more than once.

  // Make a copy as loop goups may change as we update them
  IdGroups loop_groups{
      idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets().begin(),
      idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets().end()};

  for (auto loop_group : loop_groups) {
    // Make sure the loop groups aren't promoted to multiple iter domains.
    IterDomain* promoted_id = nullptr;
    for (auto id : *loop_group) {
      auto promoted_id_it = id_promotion_map.find(id);
      if (promoted_id_it == id_promotion_map.end()) {
        continue;
      }
      if (promoted_id == nullptr) {
        promoted_id = promoted_id_it->second;
      } else {
        TORCH_INTERNAL_ASSERT(
            idGraph(IdMappingMode::ALMOSTEXACT)
                .disjointIdSets()
                .strictAreMapped(promoted_id, promoted_id_it->second),
            "Conflicting promotions found: ",
            loop_group->toString(),
            "\n  Promoted to: ",
            promoted_id->toString(),
            ", and ",
            promoted_id_it->second->toString());
      }
    }

    // If promoted id not found just grab the first ID
    if (promoted_id == nullptr) {
      promoted_id = loop_group->front();
    }
    loop_promotion_map_[loop_group] = promoted_id;
  }
}

void IterDomainGraphs::buildIndexMap(const std::vector<TensorView*>& all_tvs) {
  // Initialize map at loop leaf nodes. This needs to be done just like we
  // would in "initializeId" for the exact map. Unlike AlmostExact and
  // Permissive, index map is not a superset of exact map.
  for (auto loop_group :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    for (auto id : *loop_group) {
      auto id_disjoint_set = idGraph(IdMappingMode::INDEX)
                                 .disjointIdSets()
                                 .initializeSet(id)
                                 .first->second;

      auto def_it = id_definitions_.find(id);
      if (def_it != id_definitions_.end()) {
        auto defs = def_it->second;
        ExprGroups expr_groups;
        for (auto def : defs) {
          auto expr_set = idGraph(IdMappingMode::INDEX)
                              .disjointExprSets()
                              .initializeSet(def)
                              .first->second;
          expr_groups.pushBack(expr_set);
        }
        idGraph(IdMappingMode::INDEX).uniqueDefinitions()[id_disjoint_set] =
            expr_groups;
      } else {
        id_definitions_[id] = {};
        idGraph(IdMappingMode::INDEX).uniqueDefinitions()[id_disjoint_set] = {};
      }

      auto use_it = id_uses_.find(id);
      if (use_it != id_uses_.end()) {
        auto uses = use_it->second;
        ExprGroups expr_groups;
        for (auto use : uses) {
          auto expr_set = idGraph(IdMappingMode::INDEX)
                              .disjointExprSets()
                              .initializeSet(use)
                              .first->second;
          expr_groups.pushBack(expr_set);
        }
        idGraph(IdMappingMode::INDEX).uniqueUses()[id_disjoint_set] =
            expr_groups;
      } else {
        id_uses_[id] = {};
        idGraph(IdMappingMode::INDEX).uniqueUses()[id_disjoint_set] = {};
      }
    }
  }

  // Below is the same as building the almost exact map. It just maps through
  // trivial expressions and removes their traversal from definition/uses
  VectorOfUniqueEntries<Expr*> exprs;
  for (auto expr :
       idGraph(IdMappingMode::INDEX).disjointExprSets().disjointSets()) {
    exprs.pushBack(expr->front());
  }
  ExprGroups trivial_expr_groups;

  // Map through trivial expressions
  for (auto expr : exprs) {
    auto mapped_ids = IdGraph::isTrivialExpr(expr);
    for (auto mapped_id_group : mapped_ids) {
      for (auto id : mapped_id_group) {
        trivial_expr_groups.pushBack(
            idGraph(IdMappingMode::INDEX).disjointExprSet(expr).first);
        idGraph(IdMappingMode::INDEX).mapIds(mapped_id_group.front(), id);
      }
    }
  }

  // Clear out expressions that map inputs and outputs to the same group from
  // definitions and uses. They shouldn't be important in traversal. Iterate
  // on a copy as we're updating the map as we traverse.
  std::unordered_map<IdGroup, ExprGroups> defs_copy =
      idGraph(IdMappingMode::INDEX).uniqueDefinitions();
  for (auto& id_2_expr_group_map_entry : defs_copy) {
    ExprGroups expr_groups_new;
    for (auto& expr_group : id_2_expr_group_map_entry.second) {
      if (!trivial_expr_groups.has(expr_group)) {
        expr_groups_new.pushBack(expr_group);
      }
    }

    if (expr_groups_new.size() == id_2_expr_group_map_entry.second.size()) {
      continue;
    }

    idGraph(IdMappingMode::INDEX)
        .uniqueDefinitions()[id_2_expr_group_map_entry.first] = expr_groups_new;
  }

  std::unordered_map<IdGroup, ExprGroups> uses_copy =
      idGraph(IdMappingMode::INDEX).uniqueUses();
  for (auto& id_2_expr_group_map_entry : uses_copy) {
    ExprGroups expr_groups_new;
    for (auto expr_group : id_2_expr_group_map_entry.second) {
      if (!trivial_expr_groups.has(expr_group)) {
        expr_groups_new.pushBack(expr_group);
      }
    }

    if (expr_groups_new.size() == id_2_expr_group_map_entry.second.size()) {
      continue;
    }
    if (!expr_groups_new.empty()) {
      for (auto i : c10::irange(100)) {
        if (i > 0) {
          expr_groups_new.pushBack(expr_groups_new.front());
        }
      }
    }

    idGraph(IdMappingMode::INDEX)
        .uniqueUses()[id_2_expr_group_map_entry.first] = expr_groups_new;
  }

  for (auto loop_group :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    auto loop_promotion_it = loop_promotion_map_.find(loop_group);
  }
  IdGroups processed;

  for (auto tv : all_tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    for (auto id : tv->domain()->domain()) {
      auto loop_group_pair = idGraph(IdMappingMode::LOOP).disjointIdSet(id);
      TORCH_INTERNAL_ASSERT(
          loop_group_pair.second,
          "Loop group not found for leaf id: ",
          id->toString());
      auto loop_group = loop_group_pair.first;
      if (processed.has(loop_group)) {
        continue;
      }
      processed.pushBack(loop_group);

      auto loop_promotion_it = loop_promotion_map_.find(loop_group);
      TORCH_INTERNAL_ASSERT(loop_promotion_it != loop_promotion_map_.end());
      IterDomain* promoted_id = loop_promotion_it->second;

      for (auto loop_group_id : *loop_group) {
        if (loop_group_id == promoted_id) {
          continue;
        }
        if (idGraph(IdMappingMode::ALMOSTEXACT)
                .disjointIdSets()
                .permissiveAreMapped(loop_group_id, promoted_id)) {
          idGraph(IdMappingMode::INDEX).mapIds(loop_group_id, promoted_id);
        }
      }
    }
  }
}

ComputeAtMap::ComputeAtMap(Fusion* fusion)
    : id_graphs_(fusion), concretized_bcasts_(fusion), fusion_(fusion) {
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

    auto defs_it = id_graphs_.idGraph(IdMappingMode::ALMOSTEXACT)
                       .iterDomainGroupDefinitions(currently_visiting);
    if (!defs_it.second) {
      // TODO: Don't use ->definition()
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
  // // Scheduling can use compute at map, and may be in a bad state, only
  // check
  // // during lowering
  // if (!FusionGuard::getCurFusion()->isA<kir::Kernel>()) {
  //   return;
  // }

  // auto all_tvs = ir_utils::allTvs(FusionGuard::getCurFusion());
  // for (auto tv : all_tvs) {
  //   // Fusion inputs don't result in control flow, ignore.
  //   if (tv->isFusionInput()) {
  //     continue;
  //   }

  //   for (auto tv : all_tvs) {
  //     IdGroups tv_loop_domains;

  //     // Grab the iter domains that should be used for the for loops.
  //     VectorOfUniqueEntries<IterDomain*> loop_ids;
  //     for (auto id : tv->domain()->domain()) {
  //       // Traverse the promotion map until a leaf is found
  //       IterDomain* promoted_id = id_graphs_.getMaybePromoted(id);

  //       while (promoted_id != id_graphs_.getMaybePromoted(promoted_id)) {
  //         promoted_id = id_graphs_.getMaybePromoted(promoted_id);
  //       }

  //       TORCH_INTERNAL_ASSERT(
  //           id_graphs_.idGraph(IdMappingMode::LOOP).disjointIdSets()
  //               .mappingExists(promoted_id),
  //           "Loop id's aren't inclusive, as a producer could look to
  //           promote to an IterDomain that's not a consumer's leaf domain.",
  //           " Error from trying to promote ", id, " to ", promoted_id);
  //       auto promoted_loop_concrete_id =
  //           getConcreteMappedID(promoted_id, IdMappingMode::LOOP);

  //       loop_ids.pushBack(promoted_loop_concrete_id);
  //     }

  //     // Grab the iter domains we need to index into
  //     VectorOfUniqueEntries<IterDomain*> root_ids;
  //     for (auto id : tv->getMaybeRFactorDomain()) {
  //       if (id->isBroadcast()) {
  //         // Broadcast IDs don't need to be indexable
  //         continue;
  //       }
  //       root_ids.pushBack(id);
  //     }

  //     // // TODO: Add assert once full loop promotion is implemented.
  //     // // Check if root is indexable based on loops
  //     // TORCH_INTERNAL_ASSERT(
  //     //     indexingReachableFrom(loop_ids, root_ids),
  //     //     "Could not detect how to resolve the indexing from loop
  //     //     IterDomains: ", loop_ids.toString(), " to root iter domains:
  //     ",
  //     //     root_ids.toString(),
  //     //     "\n When checking the indexing of ",
  //     //     tv->toString());
  //   }
  // }
}

void ComputeAtMap::allocateIndexVariables() {
  // Run through all disjoint sets registered in loop map,
  //  all lowered kir::ForLoop will correspond to one of the disjoint sets
  //  and we only need one index variable for each set.
  for (const auto& loop_disjoint_set : id_graphs_.idGraph(IdMappingMode::LOOP)
                                           .disjointIdSets()
                                           .disjointSets()) {
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
      id_graphs_.idGraph(IdMappingMode::LOOP)
          .disjointIdSets()
          .mappingExists(id),
      "Index Variable: no index variable allocated as ",
      id->toString(),
      " is not registered in loop map");
  const auto* loop_set =
      id_graphs_.idGraph(IdMappingMode::LOOP).disjointIdSet(id).first.get();

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
          id_graphs_.idGraph(IdMappingMode::EXACT).disjointIdSet(disjoint_id);
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
      // Change data structure for IterDomainGraphs::buildMapBetween
      VectorOfUniqueEntries<IterDomain*> consumer_ids(
          all_consumer_ids.begin(), all_consumer_ids.end());
      for (auto producer : producers) {
        auto all_producer_ids = ir_utils::allIDsOf(producer);
        // Change data structure for IterDomainGraphs::buildMapBetween
        VectorOfUniqueEntries<IterDomain*> producer_ids(
            all_producer_ids.begin(), all_producer_ids.end());

        auto p2c = id_graphs_.idGraph(IdMappingMode::PERMISSIVE)
                       .buildMapBetween(producer_ids, consumer_ids);

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
       id_graphs_.idGraph(IdMappingMode::EXACT)
           .disjointIdSets()
           .disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    concrete_id_cache_[disjoint_set_shared_ptr] = first_id;
  }

  // The following two algorithms seem quite wasteful. Should find a more
  // efficient way to compute concrete IDs.
  for (const auto& disjoint_set_shared_ptr :
       id_graphs_.idGraph(IdMappingMode::PERMISSIVE)
           .disjointIdSets()
           .disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  // Same as exact computation
  for (const auto& disjoint_set_shared_ptr :
       id_graphs_.idGraph(IdMappingMode::ALMOSTEXACT)
           .disjointIdSets()
           .disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::ALMOSTEXACT);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graphs_.idGraph(IdMappingMode::LOOP)
           .disjointIdSets()
           .disjointSets()) {
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
  auto disjoint_sets = ca_map.idGraph(mode).disjointIdSets().disjointSets();
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

// TODO: Deduplicate with IterDomainGraphs::toString()
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
  return id_graphs_.viewRfactorIds().find(ref_id) !=
      id_graphs_.viewRfactorIds().end();
}

std::vector<IterDomain*> ComputeAtMap::getViewRfactorDomainsOfIdGroup(
    IterDomain* ref_id,
    IdMappingMode mode) const {
  auto disjoint_set = disjointSetOf(ref_id, mode);
  std::vector<IterDomain*> rfactor_ids;
  for (auto disjoint_id : disjoint_set->vector()) {
    if (id_graphs_.viewRfactorIds().find(disjoint_id) !=
        id_graphs_.viewRfactorIds().end()) {
      rfactor_ids.push_back(disjoint_id);
    }
  }
  return rfactor_ids;
}

const IdGroup ComputeAtMap::disjointSetOf(IterDomain* id, IdMappingMode mode)
    const {
  auto disjoint_set_pair = id_graphs_.idGraph(mode).disjointIdSet(id);
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
    auto defs_pair = id_graphs_.idGraph(IdMappingMode::EXACT)
                         .iterDomainGroupDefinitions(currently_visiting);

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
    auto defs_pair = id_graphs_.idGraph(IdMappingMode::EXACT)
                         .iterDomainGroupDefinitions(currently_visiting);

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
    auto uses_pair = id_graphs_.idGraph(IdMappingMode::EXACT)
                         .iterDomainGroupUses(currently_visiting);

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

void IterDomainGraphs::updateComputeWith(TensorView* compute_with_tv) {
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
          return idGraph(IdMappingMode::PERMISSIVE)
              .disjointIdSets()
              .permissiveAreMapped(id, consumer_id);
        });
    TORCH_INTERNAL_ASSERT(
        it != consumer_tv->domain()->domain().end(),
        "No consumer leaf ID of tensor ",
        consumer_tv->toString(),
        " permissively mapped with: ",
        id->toString());

    IterDomain* consumer_id = *it;

    idGraph(IdMappingMode::LOOP).mapIds(id, consumer_id);
  }
}

void ComputeAtMap::updateComputeWith(TensorView* compute_with_tv) {
  TORCH_INTERNAL_ASSERT(
      compute_with_tv->hasResolvedComputeWith(),
      "Invalid tensor: ",
      compute_with_tv->toString());

  id_graphs_.updateComputeWith(compute_with_tv);

  // Update the LOOP concrete IDs
  for (const auto& disjoint_set_shared_ptr :
       id_graphs_.idGraph(IdMappingMode::LOOP)
           .disjointIdSets()
           .disjointSets()) {
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
