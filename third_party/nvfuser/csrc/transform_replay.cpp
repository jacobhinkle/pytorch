#include <transform_replay.h>

#include <arith.h>
#include <compute_at_map.h>
#include <disjoint_set.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <maxinfo_propagator.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <deque>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

using id_map = std::unordered_map<IterDomain*, IterDomain*>;

namespace {

class ReplaySelf : public ReplayTransformations {
 private:
  // Took a good bit of this from ReplayTransformations::handle(Split...)
  void handle(Split* s) override {
    // Grab input to the split operation
    auto id_in = s->in();

    // Grab our mapping of that ID to the one we're replaying
    auto it = id_map_.find(id_in);

    // Make sure it exists in the map
    TORCH_INTERNAL_ASSERT(
        it != id_map_.end(),
        "Transform traversal failed, dependencies not met.");
    // Grab the ID we're going to replay on
    auto mapped = it->second;

    // This ID should be a leaf ID (meaning it has no uses we generated)
    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified a node but it was not a leaf node.");

    // outer loop size
    Val* remainder = ceilDiv(
        Split::extent(mapped->extent(), s->startOffset(), s->stopOffset()),
        s->factor());

    // Manually replay the split, following the output of the operations.
    // This is so rfactor ops are replayed correctly.
    IterDomain* ido =
        IterDomainBuilder(s->outer())
            .start(s->container()->zeroVal())
            .extent(s->innerSplit() ? remainder->as<Int>() : s->factor())
            .build();

    // inner IterDomain
    IterDomain* idi =
        IterDomainBuilder(s->inner())
            .start(s->container()->zeroVal())
            .extent(s->innerSplit() ? s->factor() : remainder->as<Int>())
            .build();

    // Generate the split node
    IrBuilder::create<Split>(
        s->container(),
        ido,
        idi,
        mapped,
        s->factor(),
        s->innerSplit(),
        s->startOffset(),
        s->stopOffset());

    // Remove mapped id from leaf IDs
    leaf_ids_.erase(mapped);

    // Add outputs to leaf IDs
    leaf_ids_[ido] = counter++;
    leaf_ids_[idi] = counter++;

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;
  }

  void handle(Merge* m) override {
    auto id_outer = m->outer();
    auto id_inner = m->inner();

    auto it_outer = id_map_.find(id_outer);
    auto it_inner = id_map_.find(id_inner);

    TORCH_INTERNAL_ASSERT(
        it_outer != id_map_.end() && it_inner != id_map_.end(),
        "Transform traversal failed, dependencies not met.");

    auto id_outer_mapped = it_outer->second;
    auto id_inner_mapped = it_inner->second;

    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(id_outer_mapped) != leaf_ids_.end() &&
            leaf_ids_.find(id_inner_mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified ",
        id_outer_mapped,
        " and ",
        id_inner_mapped,
        " however one or both are not leaf nodes.");

    Val* merged_id_size =
        mul(id_outer_mapped->extent(), id_inner_mapped->extent());

    IterDomain* merged_id = IterDomainBuilder(m->out())
                                .start(m->container()->zeroVal())
                                .extent(merged_id_size->as<Int>())
                                .build();

    IrBuilder::create<Merge>(
        m->container(), merged_id, id_outer_mapped, id_inner_mapped);

    // Remove inputs from the leaf IDs
    leaf_ids_.erase(id_outer_mapped);
    leaf_ids_.erase(id_inner_mapped);

    // Add the output to the leaf IDs
    leaf_ids_[merged_id] = counter++;

    id_map_[m->out()] = merged_id;
  }

 public:
  ReplaySelf(const std::vector<IterDomain*>& _target_domain, id_map _id_map)
      : ReplayTransformations(_target_domain, std::move(_id_map), false) {}
};

} // namespace

// Self replay.
TensorDomain* TransformReplay::fullSelfReplay(
    const TensorDomain* new_self_root,
    const TensorDomain* self) {
  FUSER_PERF_SCOPE("TransformReplay::fullSelfReplay");

  TORCH_INTERNAL_ASSERT(
      new_self_root->getRootDomain().size() == self->getRootDomain().size(),
      "Invalid number of IterDomains provided.");

  // Map for replay, should be pretty simple.
  id_map axis_map;
  {
    size_t i = 0;
    for (auto id : self->getRootDomain()) {
      TORCH_INTERNAL_ASSERT(
          new_self_root->getRootDomain()[i]->isReduction() ==
                  id->isReduction() &&
              new_self_root->getRootDomain()[i]->isRFactorProduct() ==
                  id->isRFactorProduct() &&
              new_self_root->getRootDomain()[i]->isBroadcast() ==
                  id->isBroadcast(),
          "Axes ",
          id,
          " and ",
          new_self_root->getRootDomain()[i],
          " do not match for self replay.");
      axis_map[id] = new_self_root->getRootDomain()[i];
      i++;
    }
  }

  // Replay producer dimensions.
  ReplaySelf replay(self->domain(), axis_map);
  std::vector<IterDomain*> new_domain(self->nDims(), nullptr);

  {
    size_t i = 0;
    for (auto id : self->domain()) {
      auto it = replay.getReplay().find(id);
      TORCH_INTERNAL_ASSERT(
          it != replay.getReplay().end(),
          "Error during replay, didn't replay an axis.");
      new_domain[i++] = it->second;
    }

    if (self->hasRFactor()) {
      std::vector<IterDomain*> new_rfactor_domain(
          self->getMaybeRFactorDomain().size(), nullptr);
      size_t i = 0;
      for (auto id : self->getMaybeRFactorDomain()) {
        auto it = replay.getReplay().find(id);
        TORCH_INTERNAL_ASSERT(
            it != replay.getReplay().end(),
            "Error during replay, didn't replay an axis.");
        new_rfactor_domain[i++] = it->second;
      }
      return IrBuilder::create<TensorDomain>(
          self->container(),
          new_self_root->getRootDomain(),
          new_rfactor_domain,
          new_domain,
          self->contiguity());
    }
  }

  return IrBuilder::create<TensorDomain>(
      self->container(),
      new_self_root->getRootDomain(),
      new_domain,
      new_self_root->contiguity());
}

namespace {

// Grab all IterDomains of producer or consumer that may not be mapped
// with consumer or producer, respectively, due to missing root
// mappings. No root mapping does not always mean dependent IDs are
// not mapped as there could be broadcast forwarded merges.
std::unordered_set<IterDomain*> getMaybeUnmappedIDs(
    const TensorView* tv,
    bool is_producer,
    const std::unordered_map<IterDomain*, IterDomain*>& root_id_map) {
  std::unordered_set<Val*> unmapped_root_ids;

  const auto& root_domain =
      is_producer ? tv->getMaybeRFactorDomain() : tv->getRootDomain();

  for (auto root_id : root_domain) {
    if (root_id_map.count(root_id) == 0) {
      unmapped_root_ids.emplace(root_id);
    }
  }

  auto all_unmapped_vals = DependencyCheck::getAllValsBetween(
      unmapped_root_ids,
      {tv->domain()->domain().begin(), tv->domain()->domain().end()});

  std::unordered_set<IterDomain*> all_unmapped_ids;
  std::transform(
      all_unmapped_vals.begin(),
      all_unmapped_vals.end(),
      std::inserter(all_unmapped_ids, all_unmapped_ids.end()),
      [](Val* val) { return val->as<IterDomain>(); });
  return all_unmapped_ids;
}

} // namespace

// Producer could have rfactor axes which consumer may want replayed. We can
// "replay" them as long as it doesn't modify the root rfactor axes. What we
// really want to do is validate if we replayed these axes to the ones they
// mapped to in the consumer the operations would all be the same. then we want
// to start the replay of the producer from the rfactor root axes, not the root.
std::pair<TensorDomain*, unsigned int> TransformReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int consumer_pos,
    const RootDomainMap& root_map,
    bool replay_swizzle) {
  FUSER_PERF_SCOPE("TransformReplay::replayPasC");
  if (producer == consumer) {
    return {producer->domain(), producer->nDims()};
  }

  if (consumer_pos < 0) {
    consumer_pos += (int)consumer->nDims() + 1;
  }

  TORCH_INTERNAL_ASSERT(
      consumer_pos >= 0 && (unsigned int)consumer_pos <= consumer->nDims(),
      "Invalid axis in transform replayPasC.");

  // consumer ids we need to match in producer
  std::vector<IterDomain*> target_consumer_ids(
      consumer->domain()->domain().begin(),
      consumer->domain()->domain().begin() + consumer_pos);

  // Instead of replaying from the root, lets try to play forward the history of
  // producer if they match ops on consumer. Enforce if we modify an rfactor
  // axis that those ops must match.
  //
  // Swizzles should not be skipped in the BestEffortReplay matching in this
  // case. If a swizzle mismatch is found, by default BestEffortReplay forwards
  // the mapping to the swizzle outputs, which would help in the case of CaMap
  // build but in the case of transform replay, would need to do the replay from
  // the inputs of the swizzles instead of the outputs, and therefore should not
  // skip swizzles in here.
  auto forward_replay = BestEffortReplay::replayPasC(
      producer, consumer, consumer_pos, root_map, false, !replay_swizzle);

  // Make a new map based on all the leaves resulting from best effort replay
  id_map forwarded_replay_map;
  auto forwarded_replay_leaves = forward_replay.getUnorderedLeafIDs();
  for (auto entry : forward_replay.getReplay()) {
    if (forwarded_replay_leaves.find(entry.second) !=
        forwarded_replay_leaves.end()) {
      forwarded_replay_map[entry.first] = entry.second;
      forwarded_replay_leaves.erase(entry.second);
    }
  }

  // Replay producer dimensions.
  ReplayTransformations replay_PasC(
      target_consumer_ids, forwarded_replay_map, false, replay_swizzle);

  auto producer_leaf_ids(replay_PasC.getUnorderedLeafIDs());

  const auto maybe_unmapped_ids = getMaybeUnmappedIDs(
      consumer,
      false,
      root_map.mapConsumerToProducer(consumer->domain(), producer->domain()));

  // Remove all ids from producer_leaf_ids that map within the consumer
  // position, we're going to try to further replay the rest of the producer
  // dimensions based on the producers original transformations. Save all dims
  // that mapped to target_consumer_ids.
  std::vector<IterDomain*> dims_mapped2target;
  for (auto c_id : target_consumer_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          maybe_unmapped_ids.count(c_id),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        producer_leaf_ids.find(it->second) != producer_leaf_ids.end(),
        "Replayed id to match consumer id ",
        c_id,
        " should be a leaf in replay map.");
    producer_leaf_ids.erase(it->second);
    dims_mapped2target.push_back(it->second);
  }

  // producer_leaf_ids now contains all producer ID products that are not used
  // to satisfy the computeAt. Put them in a replay map so we can play forward
  // these IDs in producer (if possible):
  id_map producer_self_replay_map;
  for (auto entry : producer_leaf_ids) {
    producer_self_replay_map[entry.first] = entry.first;
  }

  for (auto entry : forwarded_replay_leaves) {
    producer_self_replay_map[entry.first] = entry.first;
  }

  // Check which root domains were used to produce the producer_leaf_ids. We may
  // have picked up extra roots in consumer because of broadcast forwarding.
  std::vector<Val*> unordered_non_root_leaf_vals;
  for (auto leaf_id : replay_PasC.getUnorderedLeafIDs()) {
    if (leaf_id.first->definition() == nullptr) {
      continue;
    } else {
      unordered_non_root_leaf_vals.emplace_back(leaf_id.first);
    }
  }

  auto producer_root = producer->getMaybeRFactorDomain();

  // Figure out all id's that have been processed to generate the
  // unordered_non_root_leaf_vals. This needs to be done because we want to
  // match on producer's rfactor domain, not root domain.
  std::unordered_set<IterDomain*> all_processed_ids;
  {
    auto all_processed_vals_vec = DependencyCheck::getAllValsBetween(
        {producer_root.begin(), producer_root.end()},
        unordered_non_root_leaf_vals);
    auto all_processed_ids_vec =
        ir_utils::filterByType<IterDomain>(all_processed_vals_vec);
    all_processed_ids.insert(
        all_processed_ids_vec.begin(), all_processed_ids_vec.end());
  }

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto producer_root_id : producer_root) {
    if (all_processed_ids.find(producer_root_id) == all_processed_ids.end() &&
        std::find(
            dims_mapped2target.begin(),
            dims_mapped2target.end(),
            producer_root_id) == dims_mapped2target.end()) {
      producer_self_replay_map[producer_root_id] = producer_root_id;
    }
  }

  // Play forward transformations all producer IDs we can
  auto producer_replayed_leaves = BestEffortReplay(
      producer->domain()->domain(),
      producer->domain()->domain(),
      producer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * consumer->domain(). These are axes that were "fully replayed" relative to
   * the consumer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_leaves now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_leaves
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_leaves
   *
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  // Add axes in (1)
  for (auto c_id : target_consumer_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          maybe_unmapped_ids.count(c_id),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  unsigned int producer_pos = new_IDs.size();
  bool mismatch_found = false;

  // Add axes in (2)
  for (auto c_id : consumer->domain()->domain()) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it != replay_PasC.getReplay().end()) {
      auto id = it->second;
      // If the leaf id from ReplayTransformations is used to move
      // forward in BestEffortReplay, it is not a final ID.
      if (producer_replayed_leaves.getUnorderedLeafIDs().find(id) ==
          producer_replayed_leaves.getUnorderedLeafIDs().end()) {
        mismatch_found = true;
        continue;
      }
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
        if(!mismatch_found){
          producer_pos = new_IDs.size();
        }
      }
    }
  }

  // Add axes in (3)
  for (auto id : producer->domain()->domain()) {
    if (producer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        producer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : producer_replayed_leaves.getLeafIDs()) {
    if (used_IDs.find(id) == used_IDs.end()) {
      new_IDs.push_back(id);
    }
  }
  TensorDomain* replayed = IrBuilder::create<TensorDomain>(
      producer->container(),
      producer->getRootDomain(),
      producer->getRFactorDomain(),
      new_IDs,
      producer->domain()->contiguity());

  return {replayed, producer_pos};
}

std::pair<TensorDomain*, unsigned int> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int producer_pos,
    const RootDomainMap& root_map,
    bool replay_swizzle) {
  FUSER_PERF_SCOPE("TransformReplay::replayCasP");

  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer)
    return {consumer->domain(), consumer->nDims()};

  if (producer_pos < 0) {
    producer_pos += (int)producer->nDims() + 1;
  }

  TORCH_INTERNAL_ASSERT(
      producer_pos >= 0 && (unsigned int)producer_pos <= producer->nDims(),
      "Invalid axis in transform replayCasP. Consumer: ",
      consumer->toString(),
      " Producer: ",
      producer->toString());

  // producer ids we need to match in consumer
  std::vector<IterDomain*> target_producer_ids(
      producer->domain()->domain().begin(),
      producer->domain()->domain().begin() + producer_pos);
  target_producer_ids = TensorDomain::noReductions(target_producer_ids);

  // Instead of replaying from the root, lets try to forward the history of
  // consumer if they match ops on producer. Enforce if we modify an rfactor
  // axis that those ops match.
  //
  // Note on skip_swizzles: Similar constraints apply in replayPasC. See the
  // corresponding notes there on not skipping swizzles in the matching here.
  BestEffortReplay forward_replay = BestEffortReplay::replayCasP(
      consumer, producer, producer_pos, root_map, false, !replay_swizzle);

  // Track dangling leaves which can be produced in
  // BestEffortReplay::replayCasP these don't have any equivalent in producer
  // so they're not in the map. We will simply map them to themselves so we
  // don't lose them.
  id_map forwarded_replay_map;
  auto forwarded_replay_leaves = forward_replay.getUnorderedLeafIDs();
  for (auto entry : forward_replay.getReplay()) {
    if (forwarded_replay_leaves.find(entry.second) !=
        forwarded_replay_leaves.end()) {
      forwarded_replay_map[entry.first] = entry.second;
      forwarded_replay_leaves.erase(entry.second);
    }
  }

  // Replay producer dimensions.
  ReplayTransformations replay_CasP(
      target_producer_ids, forwarded_replay_map, replay_swizzle);

  auto consumer_leaf_ids(replay_CasP.getUnorderedLeafIDs());

  const auto maybe_unmapped_ids = getMaybeUnmappedIDs(
      producer,
      true,
      root_map.mapProducerToConsumer(producer->domain(), consumer->domain()));

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest, track all dims that are needed to match producer CA dims
  std::vector<IterDomain*> dims_mapped2target;
  for (auto p_id : target_producer_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it == replay_CasP.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          maybe_unmapped_ids.count(p_id),
          "Could not find axis, ",
          p_id,
          ", requested in replaying consumer ",
          consumer,
          " as producer ",
          producer);
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        consumer_leaf_ids.find(it->second) != consumer_leaf_ids.end(),
        "Replayed id to match producer id ",
        p_id,
        " should be a leaf in replay map.");
    consumer_leaf_ids.erase(it->second);
    dims_mapped2target.push_back(it->second);
  }

  // consumer_leaf_ids now contains all consumer ID products that are not used
  // to satisfy the computeAt. Turn into a  map so we can play forward these IDs
  // in consumer (if possible):
  id_map consumer_self_replay_map;
  for (auto entry : consumer_leaf_ids) {
    consumer_self_replay_map[entry.first] = entry.first;
  }

  for (auto entry : forwarded_replay_leaves) {
    consumer_self_replay_map[entry.first] = entry.first;
  }

  // Check which root domains were used to produce the consumer_leaf_ids. We may
  // have picked up extra roots in consumer because of broadcast forwarding.
  std::vector<Val*> unordered_non_root_leaf_vals;
  for (auto leaf_id : replay_CasP.getUnorderedLeafIDs()) {
    if (leaf_id.first->definition() == nullptr) {
      continue;
    } else {
      unordered_non_root_leaf_vals.emplace_back(leaf_id.first);
    }
  }

  auto processed_roots = IterVisitor::getInputsTo(unordered_non_root_leaf_vals);

  std::vector<IterDomain*> consumer_root = consumer->getRootDomain();

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto consumer_root_id : consumer_root) {
    if (std::find(
            processed_roots.begin(), processed_roots.end(), consumer_root_id) ==
            processed_roots.end() &&
        // Don't re-add roots that may have directly mapped in the replay
        std::find(
            dims_mapped2target.begin(),
            dims_mapped2target.end(),
            consumer_root_id) == dims_mapped2target.end()) {
      consumer_self_replay_map[consumer_root_id] = consumer_root_id;
    }
  }

  // Play forward transformations all consumer IDs we can
  auto consumer_replayed_leaves = BestEffortReplay(
      consumer->domain()->domain(),
      consumer->domain()->domain(),
      consumer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * producer->domain(). These are axes that were "fully replayed" relative to
   * the producer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_leaves now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_leaves
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_leaves
   *
   * TODO: Should (2) and (3) be swapped?
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  // Add axes in (1)
  for (auto p_id : target_producer_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it == replay_CasP.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          maybe_unmapped_ids.count(p_id),
          "Could not find axis, ",
          p_id,
          ", requested in replay.");
      continue;
    }
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  // Add axes in (2)
  for (auto p_id : producer->domain()->domain()) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it != replay_CasP.getReplay().end()) {
      auto id = it->second;
      // If the leaf id from ReplayTransformations is used to move
      // forward in BestEffortReplay, it is not a final ID.
      if (consumer_replayed_leaves.getUnorderedLeafIDs().find(id) ==
          consumer_replayed_leaves.getUnorderedLeafIDs().end()) {
        continue;
      }
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // This position doesn't quite match the position inliner would considered
  // "replayed at". Both (1) and (2) aren't quite right as a reduction dim in
  // producer would be "maybe unampped" however we can't inline relative to it.
  unsigned int consumer_pos = new_IDs.size();

  // Add axes in (3)
  for (auto id : consumer->domain()->domain()) {
    if (consumer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        consumer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : consumer_replayed_leaves.getLeafIDs()) {
    if (used_IDs.find(id) == used_IDs.end()) {
      new_IDs.push_back(id);
    }
  }

  TensorDomain* replayed = IrBuilder::create<TensorDomain>(
      consumer->container(),
      consumer->getRootDomain(),
      consumer->getRFactorDomain(),
      new_IDs,
      consumer->domain()->contiguity());

  return {replayed, consumer_pos};
}

// replay Producer as Consumer
std::pair<TensorDomain*, unsigned int> TransformReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int compute_at_axis,
    bool replay_swizzle) {
  // Use the pairwise root map as a default mapper
  PairwiseRootDomainMap root_map(producer, consumer);
  return replayPasC(
      producer, consumer, compute_at_axis, root_map, replay_swizzle);
}

std::pair<TensorDomain*, unsigned int> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int compute_at_axis,
    bool replay_swizzle) {
  // Use the pairwise root map as a default mapper
  PairwiseRootDomainMap root_map(producer, consumer);
  return replayCasP(
      consumer, producer, compute_at_axis, root_map, replay_swizzle);
}

namespace {
bool isProducerOf(
    const TensorView* maybe_producer,
    const TensorView* maybe_consumer) {
  if (maybe_consumer->definition() == nullptr) {
    return false;
  }
  auto def = maybe_consumer->definition();
  for (auto inp : ir_utils::filterByType<TensorView>(def->inputs())) {
    if (maybe_producer == inp) {
      return true;
    }
  }

  return false;
}

bool isSiblingOf(
    const TensorView* maybe_sibling_0,
    const TensorView* maybe_sibling_1) {
  if (maybe_sibling_0->definition() == nullptr) {
    return false;
  }
  auto def = maybe_sibling_0->definition();
  for (auto other_output_tv :
       ir_utils::filterByType<TensorView>(def->outputs())) {
    if (other_output_tv == maybe_sibling_1) {
      return true;
    }
  }

  return false;
}
} // namespace

// Return the position in target that matches with reference at maximum position
// reference_pos
int TransformReplay::getMatchedLeafPosWithoutReplayTasR(
    const TensorView* target,
    const TensorView* reference,
    int reference_pos) {
  FUSER_PERF_SCOPE("transform_replay.cpp::getMatchedLeafPosWithoutReplayTasR");

  if (reference_pos < 0) {
    reference_pos += reference->nDims();
  }

  TORCH_INTERNAL_ASSERT(
      reference_pos >= 0 && reference_pos <= reference->nDims(),
      reference_pos,
      " is an invalid posiotion for ",
      reference->toString());

  Expr* definition_to_map = nullptr;

  std::vector<IterDomain*> target_root;
  std::vector<IterDomain*> reference_root;

  // Some logic still dependent on if producer or consumer (i.e. PasC vs CasP)
  //
  // Would be nice if this was concisely captured in the IterDomainGraph
  const TensorView* producer = nullptr;
  const TensorView* consumer = nullptr;

  if (isProducerOf(reference, target)) {
    // CasP
    consumer = target;
    producer = reference;

    definition_to_map = target->definition();
    reference_root = reference->getMaybeRFactorDomain();
    target_root = target->getRootDomain();
  } else if (isProducerOf(target, reference)) {
    // PasC
    producer = target;
    consumer = reference;

    definition_to_map = reference->definition();
    reference_root = reference->getRootDomain();
    target_root = target->getMaybeRFactorDomain();
  } else if (target == reference) {
    return (int)target->domain()->nDims();
  } else if (isSiblingOf(target, reference)) {
    reference_root = reference->getRootDomain();
    target_root = target->getRootDomain();
    definition_to_map = target->definition();
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Unsupported relationship for",
        " getMatchedLeafPosWithoutReplayTasR with reference: ",
        reference->toString(),
        ", and ",
        target->toString());
  }

  IterDomainGraph id_graph({definition_to_map});

  auto r2t_permissive_map = id_graph.buildMapBetween(
      ir_utils::allIDsOf(reference),
      ir_utils::allIDsOf(target),
      IdMappingMode::PERMISSIVE);

  // The only dimensions we can actually skip in the replay is consumer
  // broadcast dimensions that don't map to any dimensions in producer.
  VectorOfUniqueEntries<IterDomain*> skippable_root_dims;
  if (consumer != nullptr) {
    for (auto c_root_id : consumer->getRootDomain()) {
      if (c_root_id->isBroadcast()) {
        skippable_root_dims.pushBack(c_root_id);
      }
    }
    for(auto r2t_entry : r2t_permissive_map){
      auto r_id = r2t_entry.first;
      if(r2t_entry.second.empty()){
        continue;
      }
      skippable_root_dims.erase(r_id);
      for(auto t_id : r2t_entry.second){
        skippable_root_dims.erase(t_id);
      }
    }
  }

  if (producer != nullptr) {
    for (auto p_root_id : producer->getMaybeRFactorDomain()) {
      if (p_root_id->isBroadcast()) {
        skippable_root_dims.pushBack(p_root_id);
      }
    }
    for(auto r2t_entry : r2t_permissive_map){
      auto r_id = r2t_entry.first;
      if(r2t_entry.second.empty()){
        continue;
      }
      skippable_root_dims.erase(r_id);
      for(auto t_id : r2t_entry.second){
        skippable_root_dims.erase(t_id);
      }
    }
  }
  
  VectorOfUniqueEntries<IterDomain*> unskippable_root_dims;
  for(auto r_root_id : reference_root){
    if(!skippable_root_dims.has(r_root_id)){
      unskippable_root_dims.pushBack(r_root_id);
    }
  }

  for(auto t_root_id : target_root){
    if(!skippable_root_dims.has(t_root_id)){
      unskippable_root_dims.pushBack(t_root_id);
    }
  }

  VectorOfUniqueEntries<IterDomain*> unskippable_domain_ids;

  const auto target_domain = target->domain()->domain();
  const auto reference_domain = reference->domain()->domain();

  {
    std::vector<IterDomain*> target_reference_domains = target_domain;
    target_reference_domains.insert(
        target_reference_domains.begin(),
        reference_domain.begin(),
        reference_domain.end());

    auto unskippable_ids_vec = DependencyCheck::getAllValsBetween(
        {unskippable_root_dims.vector().begin(),
         unskippable_root_dims.vector().end()},
        {target_reference_domains.begin(), target_reference_domains.end()});

    std::unordered_set<Val*> unskippable_ids_set(
        {unskippable_ids_vec.begin(), unskippable_ids_vec.end()});

    for (auto id : target_reference_domains) {
      if (unskippable_ids_set.find(id) != unskippable_ids_set.end()) {
        unskippable_domain_ids.pushBack(id);
      }
    }
  }

  auto it_reference = reference_domain.begin();
  auto it_target = target_domain.begin();

  while ((it_reference != reference_domain.end() ||
          it_target != target_domain.end()) &&
         (int)std::distance(reference_domain.begin(), it_reference) !=
             reference_pos) {
    if (it_target != target_domain.end()) {
      auto target_id = *it_target;
      if (!unskippable_domain_ids.has(target_id)) {
        ++it_target;
        continue;
      }
    }

    if (it_reference != reference_domain.end()) {
      auto reference_id = *it_reference;
      if (!unskippable_domain_ids.has(reference_id)) {
        ++it_reference;
        continue;
      }
    }

    if (it_reference == reference_domain.end() ||
        it_target == target_domain.end()) {
      break;
    }

    auto reference_id = *it_reference;
    auto target_id = *it_target;

    if (id_graph.getDisjointIdSets(IdMappingMode::PERMISSIVE)
            .permissiveAreMapped(reference_id, target_id)) {
      ++it_reference;
      ++it_target;
      continue;
    }

    break;
  }

  if ((int)std::distance(reference_domain.begin(), it_reference) ==
      reference_pos) {
    return (int)std::distance(target_domain.begin(), it_target);
  } else {
    return -1;
  }
}

namespace {

// Make sure if tv is set to new_td it doesn't violate set compute at and max
// produce at positions.
bool validateDomain(TensorView* tv, TensorDomain* new_td) {
  auto first_mismatch =
      BestEffortReplay::findFirstMismatchedID(tv->domain(), new_td);
  return first_mismatch >= (int)tv->getMaybeMaxProducerPosition() &&
      first_mismatch >= (int)tv->getMaxComputePosition();
}

} // namespace

void TransformPropagator::propagateC2P(TensorView* from, TensorView* to) {
  int pos = replayed_pos_.at(from);
  // Note: [Using multiple TransformPropagators]
  // There are cases that we use multiple TransformPropagators along different
  // spanning trees with different references in the same fusion. Some of these
  // spanning trees could overlap. In cases when there are overlapping nodes,
  // TransformPropagator needs to respect the replay of others, because the
  // current TransformPropagator might not contain the most amount of
  // information on how to do the correct transformation. The logic below tells
  // TransformPropagator to skip the replay when not necessary.
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayTasR(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "TransformPropagator::propagateC2P" << std::endl;
    std::cout << "  from: " << from << " @ " << pos << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayPasC(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    new_pos = replay.second;
    if (debug) {
      std::cout << "  replayed: " << to << " @ " << new_pos << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped. result position: " << new_pos << std::endl;
  }
  replayed_pos_[to] = new_pos;
}

void TransformPropagator::propagateP2C(TensorView* from, TensorView* to) {
  int pos = replayed_pos_.at(from);
  // See note [Using multiple TransformPropagators]
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayTasR(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "TransformPropagator::propagateP2C" << std::endl;
    std::cout << "  from: " << from << " @ " << pos << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayCasP(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    new_pos = replay.second;
    if (debug) {
      std::cout << "  replayed: " << to << " @ " << new_pos << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped. result position: " << new_pos << std::endl;
  }
  replayed_pos_[to] = new_pos;
}

void TransformPropagator::propagateSibling(TensorView* from, TensorView* to) {
  int pos = replayed_pos_.at(from);
  // See note [Using multiple TransformPropagators]
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "TransformPropagator::propagateSibling" << std::endl;
    std::cout << "  from: " << from << " @ " << pos << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (TransformReplay::getMatchedLeafPosWithoutReplayTasR(to, from, -1) == -1) {
    auto replay = TransformReplay::fullSelfReplay(to->domain(), from->domain());
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay),
        "Tried to set the domain of ",
        to,
        " to ",
        replay,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay);
    if (debug) {
      std::cout << "  replayed: " << to << " @ " << pos << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped. result position: " << pos << std::endl;
  }
  replayed_pos_[to] = pos;
}

TransformPropagator::TransformPropagator(TensorView* from, int64_t pos) {
  if (pos < 0) {
    pos += int64_t(from->nDims()) + 1;
  }
  TORCH_CHECK(
      pos >= 0 && pos <= (int64_t)from->nDims(),
      "TransformPropagator called on an pos outside valid range.");
  replayed_pos_[from] = pos;
}

void MostInlinedTransformPropagator::propagateC2P(
    TensorView* from,
    TensorView* to) {
  int pos = from->nDims();
  // See note [Using multiple TransformPropagators]
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayTasR(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "MostInlinedTransformPropagator::propagateC2P" << std::endl;
    std::cout << "  from: " << from << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayPasC(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    if (debug) {
      std::cout << "  replayed: " << to << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped" << std::endl;
  }
}

void MostInlinedTransformPropagator::propagateP2C(
    TensorView* from,
    TensorView* to) {
  int pos = from->nDims();
  // See note [Using multiple TransformPropagators]
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayTasR(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "MostInlinedTransformPropagator::propagateP2C" << std::endl;
    std::cout << "  from: " << from << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayCasP(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    if (debug) {
      std::cout << "  replayed: " << to << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped" << std::endl;
  }
}

void MostInlinedTransformPropagator::propagateSibling(
    TensorView* from,
    TensorView* to) {
  // See note [Using multiple TransformPropagators]
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "MostInlinedTransformPropagator::propagateSibling"
              << std::endl;
    std::cout << "  from: " << from << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (TransformReplay::getMatchedLeafPosWithoutReplayTasR(to, from, -1) == -1) {
    auto replay = TransformReplay::fullSelfReplay(to->domain(), from->domain());
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay),
        "Tried to set the domain of ",
        to,
        " to ",
        replay,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay);
    if (debug) {
      std::cout << "  replayed: " << to << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped" << std::endl;
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
