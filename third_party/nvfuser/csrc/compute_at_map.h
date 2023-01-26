#pragma once

#include <disjoint_set.h>
#include <ir_all_nodes.h>
#include <kernel_ir.h>
#include <lower_trivial_broadcast.h>

#include <deque>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

using IdGroup = std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>;
using IdGroups = VectorOfUniqueEntries<IdGroup>;
using ExprGroup = std::shared_ptr<VectorOfUniqueEntries<Expr*>>;
using ExprGroups = VectorOfUniqueEntries<ExprGroup>;

// TODO: Remove, used for IdGraph friend access.
class ComputeAtMap;

// There's three modes of these iter domain mappings all uniquely important in
// the lowering process.
//
// For EXACT/PERMISSIVE mode consider:
//
// consumer[i0, b1] = producer[i0]
// consumer->merge(0) (consumer will now be [i0 * b1])
// When producer is replayed as consumer (the direction we use for mapping)
// with BestEffortReplay forward_bcast_mismatch = True the producer to
// consumer map will have both a mapping of consumer(i0) to producer(i0) as
// well as consumer(i0*b1) to producer(i0). This latter mapping is important
// for loop nest mappings as the consumer will generate a loop based on i0*b1
// and the producer may be computeAt inside this loop nest. However, for
// indexing we do not want these two maps as producer may be indexed as i0*i1
// depending on the loop nest structure and how it was built. Therefore we
// really need to carry (at least) two sets of maps around for lowering.
//
// LOOP mode is important if we have something like:
// consumer[i0o, threadIdx.x{i0i}] = producer[i0o, threadIdx.y{i0i}](computeAt
// = 1) which can easily happen when using shared memory. We want to make sure
// that the iteration domain used for loop construction (concreteId) has the
// proper parallelization strategy. In parallel mode we do typical iteration
// domain mapping, however we remove from it any iteration domains outside the
// computeAt of producer when mapping. This guarentees we won't map
// IterDomains that could have different parallelization strategies. We also
// propagate the parallel strategy in parallel mode so all mapped IDs that
// must have the same parallel type, do.
//
// IdMappingMode::LOOP
//   Only maps leaf axes to left of compute at
//   Forward broadcast axes in replay
// IdMappingMode::PERMISSIVE
//   Forward broadcast axes in replay
//   Map all iteration domains
//   Always contain root mappings (otherwise they could have been forwarded in
//   broadcast)
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
// IdMappingMode::AlmostExact
//   Forward through broadcast axes, but not through to a non-broadcast axis
//     i.e. id{b1*i0}, id{i0} are mapped
//          id{i1*i0}, id{i0} are not mapped (this part is the difference from
//          PERMISSIVE)
//   Forward through split one axes, i.e. id{ceilDiv(i0, 1)}, id{i0} are mapped
//
class TORCH_CUDA_CU_API IterDomainGraph {
 public:
  IterDomainGraph(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs,
      bool allow_self_mapping = false);

  IterDomainGraph(
      const std::vector<Expr*>& exprs,
      bool allow_self_mapping = false);

  // Same as the above constructor with fusion->exprs() excpet fusion may have
  // some dangling inputs/outputs that are expected to have IterDomain entries
  // even though there's no possible connections from them.
  IterDomainGraph(Fusion* fusion, bool allow_self_mapping = false);

  // Returns the disjoint set according to one of the mapping mode types.
  const DisjointSets<IterDomain*>& getDisjointIdSets(IdMappingMode mode) const;

  // Returns
  //   {
  //     (1) The disjoint set of the provided Iter Domain in the provided
  //     mapping
  //       mode if it exists, otherwise a null shared ptr
  //     (2) If the disjoint set of the provided Iter Domain in the proivded
  //       mapping mode exists
  //   }
  std::pair<IdGroup, bool> getDisjointIdSet(IterDomain* id, IdMappingMode mode)
      const;

  // Returns the disjoint set according to one of the mapping mode types.
  const DisjointSets<Expr*>& getDisjointExprSets(IdMappingMode mode) const;

  // Same as getDisjointIdSet but for the Expression sets.
  std::pair<ExprGroup, bool> getDisjointExprSet(Expr* expr, IdMappingMode mode)
      const;

  // IterDomains from the original fusion are only allowed to be used once in
  // the IterDomain graph, id->uses() are not directly used as there's no bounds
  // check that would prevent a use from being defined that's not part of the
  // actual fusion definition.
  //
  // Note, any iter domains used during something like loop or concrete id
  // resolution could actually have multiple Expr* uses, and uses on disjoint id
  // sets should be used, not this.
  //
  // TODO: Can this be private?
  Expr* idUse(IterDomain* id) const;

  // TODO: Seems a bit unfortunate that this isn't IterDomain local information.
  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

  // Returns if a self mapping was detected that would invalidate assumptions of
  // the overall lowering system.
  //
  // TODO: Can we make this more of an alias analysis?
  // Ref: https://github.com/csarofeen/pytorch/pull/1954#discussion_r961940498
  bool hasSelfMapping() const {
    return self_mapping_info_.has_value();
  }

  // Convert unique vector of expressions to unique vector of it's groups in
  // provided mode
  ExprGroups toGroups(
      const VectorOfUniqueEntries<Expr*>& exprs,
      IdMappingMode mode) const;

  // Convert unique vector of IterDomain to unique vector of it's groups in
  // provided mode
  IdGroups toGroups(
      const VectorOfUniqueEntries<IterDomain*>& ids,
      IdMappingMode mode) const;

  // Return input iter domain groups of provided expr in provided mode
  IdGroups outputGroups(ExprGroup expr, IdMappingMode mode) const;

  // Return output iter domain groups of provided expr in provided mode
  IdGroups inputGroups(ExprGroup expr, IdMappingMode mode) const;

  // Traverses uses of the IterDomains in 'of' and returns all IterDomain
  // groups that depend on them in provided mapping mode.
  ExprGroups allUsesOf(const IdGroups& of, IdMappingMode mode) const;

  // Traverses definitions of the IterDomains in 'of' and returns all IterDomain
  // groups 'of' IterDomains depend on in provided mapping mode.
  ExprGroups allDefinitionsOf(const IdGroups& of, IdMappingMode mode) const;

  // Return sorted expressions to go from the provided IterDomains in from to
  // the provided IterDomains in to with provided mode. Minimal expressions to
  // get from 'from' to 'to' returned.
  ExprGroups getExprsBetween(
      const IdGroups& from,
      const IdGroups& to,
      IdMappingMode mode) const;

  // Update the LOOP ID disjoint sets with resolved computeWith
  void updateComputeWith(TensorView* compute_with_tv);

  // Supports one to many mappings, uses the disjoint sets of the provided mode
  // to produce mappings between from and to. If multiple IterDomains in to map
  // to a single iter domain in from, the order of the IterDomains in value of
  // the map is preserved to be the order provided in to.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
  buildMapBetween(
      const std::vector<IterDomain*>& from,
      const std::vector<IterDomain*>& to,
      IdMappingMode mode) const;

  // Alias of the above on unique vector entries
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
  buildMapBetween(
      const VectorOfUniqueEntries<IterDomain*>& from,
      const VectorOfUniqueEntries<IterDomain*>& to,
      IdMappingMode mode) const;

  //! Returns
  //!   (1) The expressions associated with the definitions of the provided
  //!     IterDomain group in the provided mapping mode (if it exists).
  //!   (2) If there is a definitions entry of the provided IterDomain group in
  //!     the provided mapping mode.
  //! First entry in the returned pair is a vector of vector of expressions. The
  //! inner vector is proven to be equivalent based on the provided mode. The
  //! outer vector are expression groups that are not equivalent based on the
  //! provided mode, but produce one of the IterDomains within the same disjoint
  //! Iter Domain set based on the provided mode.
  //! TODO: Change name to start with get
  std::pair<ExprGroups, bool> getIterDomainGroupDefinitions(
      IdGroup id_group,
      IdMappingMode mode) const;

  //! Same as getIterDomainGroupDefinitions but for uses instead of definitions
  //! TODO: Change name to start with get
  std::pair<ExprGroups, bool> getIterDomainGroupUses(
      IdGroup id_group,
      IdMappingMode mode) const;

  std::string toString() const;

  auto getMaybePromoted(IterDomain* id) {
    auto loop_entry_it = loop_promotion_map_.find(id);
    if (loop_entry_it != loop_promotion_map_.end()) {
      return loop_entry_it->second;
    }
    return id;
  }

  // Replay Expr but with the inputs provided. Input mapping will set a pairwise
  // mapping between new_inputs and expr->inputs()
  Expr* addReplayAs(
      const std::vector<IterDomain*>& new_inputs,
      Expr* expr,
      IdMappingMode input_mapping);

  // TODO: Remove protected, doing this now so compute at map can extend the
  // iter domain graph.
 protected:
  friend ComputeAtMap;
  // Sometimes fusion inputs or outputs are disconnected from expressions, in
  // those cases we still may want to send in some additional tensor views from
  // the Fusion that don't have expressions associated with them.
  void build(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs);

  // Copies all information computed for from into to. Useful for incremental
  // building of graph without having to rebuild entire graphs under a new mode.
  void copyGraph(IdMappingMode from_mode, IdMappingMode to_mode);

  // ======= START Iteration domain build process in order called =======

  // Fills id_uses_ for all IterDomains active in the fusion.
  void buildIterDomainUses(const std::vector<TensorView*>& all_tvs);

  // Initializes entries for the provided IterDomain in the overall
  // IterDomainGraph
  void initializeId(IterDomain* id, bool is_view_rfactor_id, bool is_leaf_id);

  // Iterates over all IterDomains in allTvs(fusion) computes
  // is_view_rfactor_id, is_leaf_id and calls initializeID.
  void initialIdProcessing(const std::vector<TensorView*>& all_tvs);

  // Map through loop swizzles, as input/output IterDomains are exact, only the
  // order they're traversed differs.
  void mapThroughLoopSwizzles(IdMappingMode mode);

  // Fills disjoint_ids_[IdMappingMode::EXACT] for relationships between inputs
  // and first output of expr
  void buildExactMap(const std::vector<Expr*>& exprs);

  // Fills disjoint_ids_[IdMappingMode::ALMOSTEXACT]. Initialize AlmostExact as
  // Exact entries, then map anything that's either merged with a size-1 or
  // split by a size-1 dimension.
  void buildAlmostExactMap();

  // Fills disjoint_ids_[IdMappingMode::PERMISSIVE]. Initialize PermissiveMap as
  // AlmostExact entries, then map through broadcasts
  void buildPermissiveMap(const std::vector<Expr*>& exprs);

  // Fills disjoint_ids_[IdMappingMode::LOOP] for relationships between inputs
  // and first output of expr
  void buildLoopMap(const std::vector<Expr*>& exprs);

  void buildLoopPromotionMap();

  // ======= END Iteration domain build process in order called =======

  // Non-const internal only version of getDisjointIdSets.
  DisjointSets<IterDomain*>& disjointIdsSet(IdMappingMode mode);

  // Non-const internal only version of getDisjointExprsSet.
  DisjointSets<Expr*>& disjointExprsSet(IdMappingMode mode);

  // Returns if first and second are expressions through which the provided
  // id_map have matching inputs (if forward), or outputs (if not forward).
  // Returning true means the expressions are "the same", in terms they modify
  // matching original extents, by the same amount.
  bool exprsMap(Expr* first, Expr* second, bool forward, IdMappingMode mode)
      const;

  // Set id0 and id1 to mapped in disjointIdsSet[mode], update id0->definition()
  // and id1->definition() sets in disjointExprsSet.
  void mapIds(IterDomain* id0, IterDomain* id1, IdMappingMode mode);

  // Checks if expr's are considered "the same" where sameness inputs and
  // outputs in the same position across expressions map with  provided
  // MappingMode. If the expressions are determined the same then
  // if forward
  //   will map outputs
  // else
  //   will map inputs
  // in the provided mode.
  // Returns if expressions were mapped through.
  bool mapThroughExpr(
      Expr* first,
      Expr* second,
      bool forward,
      IdMappingMode mode);

  // Errors if self mapping occurs
  void assertNoSelfMapping();

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  std::unordered_map<IdMappingMode, DisjointSets<IterDomain*>> disjoint_ids_;

  // Keeps a disjoint set entry for all Expressions for all mapping mode types.
  std::unordered_map<IdMappingMode, DisjointSets<Expr*>> disjoint_exprs_;

  std::unordered_map<IdMappingMode, std::unordered_map<IdGroup, ExprGroups>>
      unique_definitions_;

  std::unordered_map<IdMappingMode, std::unordered_map<IdGroup, ExprGroups>>
      unique_uses_;

  // If multiple transformations occur IterDomains could have multiple uses,
  // however only one should be active in the given Fusion. Track what the
  // active IterDomain uses are, they can only be used once.
  std::unordered_map<IterDomain*, Expr*> id_uses_;

  // Hold a set of IterDomains that are considered view rfactor ids. This
  // identification is particularly important to understand if split operations
  // are divisible or not.
  std::unordered_set<IterDomain*> view_rfactor_ids_;

  // Debug information to hold if a self mapping in a TensorView is found.
  c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
      self_mapping_info_ = c10::nullopt;

  std::unordered_map<IterDomain*, IterDomain*> loop_promotion_map_;
};

using DoubleBufferIndices = std::unordered_map<DoubleBufferLoopStage, Int*>;

class TORCH_CUDA_CU_API ComputeAtMap {
 public:
  ComputeAtMap() = delete;
  ComputeAtMap(const ComputeAtMap&) = delete;
  ComputeAtMap& operator=(const ComputeAtMap&) = delete;
  ComputeAtMap(ComputeAtMap&&) = default;
  ComputeAtMap& operator=(ComputeAtMap&&) = default;
  ComputeAtMap(Fusion* fusion);

  //! Run through disjoint sets in the LOOP map, make sure there's only one
  //! non-serial parallel type in each disjoint set, set the parallel type of
  //! all IterDomains in the disjoint set to that PType.
  //!
  //! TODO: Should this be moved to parallel validation?
  void validateAndPropagatePType();

  //! Run through disjoint sets in the LOOP map and allocate the index
  //!  variable for the associated for loop that will be generated
  //!  for each disjoint sets in the loop map. This pre-allocation makes
  //!  2 key assumptions about computeAt map that would very likely be
  //!  long term invariant:
  //!    1. All kir::forloop created in the lowering pass should belong
  //!  to one of the disjoint sets in loop map.
  //!    2. The lowering pass will *never* create a loop nest with 2
  //!  different nesting levels mapped together, i.e. the case below
  //!  never occurs:
  //!   for i in IterDomain1
  //!    for j in IterDomain2
  //!     ...
  //!   With loop_map.areMapped(IterDomain1, IterDomain2) == true.
  //! Under this condition, we can pre-allocate all required index
  //!  variable integers before creating any kir::forloop, and this
  //!  would help optimizing the generated integer math for indexing.
  //!
  //! TODO: Should this be moved to an indexing map structure outside of
  //! ComputeAtMap that has a ComputeAtMap reference?
  void allocateIndexVariables();

  //! Simple alias to IdGraph mappings.
  bool areMapped(IterDomain* id0, IterDomain* id1, IdMappingMode mode) const {
    return idGraph().getDisjointIdSets(mode).strictAreMapped(id0, id1);
  }
  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return IterDomains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id, IdMappingMode mode) const;

  // Prints mapping information, forwards to an internal IterDomainGraph
  std::string toString() const;

  // Returns if the provided ID is a view like rfactor id
  bool isViewRfactor(IterDomain* ref_id) const;

  // Returns all rfactor domains in rfactor_concrete_count_reset_domains_ that
  // are in the disjoint set of the provided IterDomain. This will be every view
  // like rfactor ID the provided ID "depends" on in the map.
  std::vector<IterDomain*> getViewRfactorDomainsOfIdGroup(
      IterDomain* ref_id,
      IdMappingMode mode) const;

  const IterDomainGraph& idGraph() const {
    return id_graph_;
  }

  //! Returns the pre-allocated index variable integer used in
  //!  the kir::ForLoop corresponding to the given IterDomain.
  //!  this interface is only valid if the ID has a loop mapping,
  //!  ca_map will throw exceptions if given iterdomain doesn't
  //!  have a loop map entry.
  Val* getIndexVariable(
      IterDomain* id,
      DoubleBufferLoopStage double_buffer_loop_stage =
          DoubleBufferLoopStage::NotApplicable) const;

  // Simple alias to IterDomainGraph::getDisjointIdSet
  const IdGroup disjointSetOf(IterDomain* id, IdMappingMode mode) const;

  // Update the LOOP map with resolved computeWith
  void updateComputeWith(TensorView* compute_with_tv);

 private:
  // Traverses through definitions of exact maps (unique_exact_definitions_) to
  // input ID's from provided ID. Returns all the exact map concrete IDs of the
  // exact sets that are inputs required to construct the exact concrete id of
  // of_id.
  IdGroups getInputDisjointSetsOf(IdGroup of_id, bool stop_at_rfactor = true);

  // Starts at exact_sets, traverses through defintions of the exact map to
  // all terminating input ID's. Returns all the exact mapped groups of all the
  // on these paths including the exact_sets.
  IdGroups getAllDisjointSetProducers(const IdGroups& exact_sets);

  // Starts at exact_sets, traverses through uses of the exact map to
  // all terminating output ID's. Returns all the exact mapped groups of all the
  // on these paths including the exact_sets.
  IdGroups getAllDisjointSetConsumers(const IdGroups& exact_sets);

  // Build id_graph_
  void build(Fusion* fusion);

  // Compute the concrete Id assocaited with id in provided mode and add its
  // entry entry in  concrete_cache_id_
  IterDomain* computeConcreteId(IterDomain* id, IdMappingMode mode);

  // TODO: remove or reimplemnt
  void buildConsumersMap();

  // TODO: Rename to computeConcreteIds
  void buildConcreteIds();

  // Temporary pass to make sure loop promotion is working as anticipated. May
  // want to keep this as validation, but also may want to remove it.
  void testValidate();

  // Considering the DAG:
  //   Inputs defined as the Almost Exact sets for IterDomains in from
  //   Outputs defined as the Almost Exact sets for IterDomains in to
  //   Directed edges as unique_exact_definitions_
  // Return if the DAG has all inputs to reach all outputs
  bool indexingReachableFrom(
      const VectorOfUniqueEntries<IterDomain*>& from,
      const VectorOfUniqueEntries<IterDomain*>& to);

  // Should be built once and never modified again.
  IterDomainGraph id_graph_;

  // Used specifically for concrete ID computation
  ConcretizedBroadcastDomains concretized_bcasts_;

  // Prevent needing to recompute concrete_id's in compute at map.
  // VectorOfUniqueEntries is unique across mapping modes, so don't need to use
  // mapping mode directly in this cache. const
  // VectorOfUniqueEntries<IterDomain*>& is what's returned by
  // ComputeAtMap::disjointSetOf which can be used directly.
  std::unordered_map<IdGroup, IterDomain*> concrete_id_cache_;

  // Permissive based map, input is a producer IterDomain and output is a list
  // of IterDomains in producer's consumers that permissively map. Primarily
  // used for concrete IterDomain resolution.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      consumers_map_;

  //! Allocated Loop index variable through the CA map.
  //!   only valid for disjoint sets on the loop ca map.
  std::unordered_map<const VectorOfUniqueEntries<IterDomain*>*, Val*>
      loop_index_variable_map_;

  //! Allocated loop indices for double buffer loop.
  //!  only valid for disjoint sets on the loop ca map
  //!  that have double buffer-ed iterdomains.
  using DoubleBufferIndicesPtr = std::unique_ptr<DoubleBufferIndices>;
  std::unordered_map<
      const VectorOfUniqueEntries<IterDomain*>*,
      DoubleBufferIndicesPtr>
      double_buffered_loop_index_variable_map_;

  // Shortcut to access the fusion this computeAt map was
  //  built from.
  Fusion* fusion_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
