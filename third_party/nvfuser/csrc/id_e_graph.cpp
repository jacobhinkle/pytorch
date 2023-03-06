#include "id_e_graph.h"

#include <deque>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void IterDomainEGraph::initGraph() {
  // Initialize a partition of all IterDomains in the Fusion, initially
  // containing all singleton classes.
  for (auto v : fusion_.vals()) {
    if (v->getValType() == ValType::IterDomain) {
      auto id = reinterpret_cast<IterDomain*>(v);
      all_ids_.push_back(id);
      all_extents_.insert(id->extent());
    }
  }
  id_partition_ =
      std::unique_ptr<UnionFind<IterDomain*>>(new UnionFind(all_ids_));
  extent_partition_ =
      std::unique_ptr<UnionFind<Val*>>(new UnionFind(all_extents_));

  for (auto expr : fusion_.unordered_exprs()) {
    // Expressions fall into one of the following categories:
    //   Broadcast
    //   Reduction
    //   Reshape
    //   Permute
    //   Pointwise
    if (expr->isA<BroadcastOp>()) {
      std::cout << "Broadcast op: " << expr->toString() << std::endl;
      auto bop = reinterpret_cast<BroadcastOp*>(expr);
      auto bcast_flags = bop->getBroadcastDimFlags();
      auto inp = reinterpret_cast<TensorView*>(bop->in());
      auto outp = reinterpret_cast<TensorView*>(bop->out());
      auto indom = inp->domain()->noReductions();
      auto outdom = outp->domain()->noReductions();
      for (size_t i = 0, j = 0; i < bcast_flags.size(); ++i) {
        if (bcast_flags[i]) {
          // This output dim is a new bcast dimension
          continue;
        }
        auto idin = indom[j++];
        auto idout = outdom[i];
        id_partition_->mergeSetsFromValues(idin, idout);
        extent_partition_->mergeSetsFromValues(idin->extent(), idout->extent());
      }
    } else if (expr->isA<ReductionOp>()) {
      std::cout << "Reduction op: " << expr->toString() << std::endl;
      auto rop = reinterpret_cast<ReductionOp*>(expr);
      // Instead of flags, we just look at the output type to find reduction
      // IDs, which will tell us which axes are being reduced over
      auto inp = reinterpret_cast<TensorView*>(rop->in());
      auto outp = reinterpret_cast<TensorView*>(rop->out());
      auto indom = inp->domain()->domain();
      auto outdom = outp->domain()->domain();
      for (size_t i = 0; i < indom.size(); ++i) {
        auto idin = indom[i];
        auto idout = outdom[i];
        if (idout->isReduction()) {
          // Don't merge serial input with rdomain output
          // TODO: set REDUCES relation
          id_relations_.push_back(Relation(RelationType::Reduces, idin, idout));
          continue;
        }
        if (idin->isBroadcast()) {
          id_relations_.push_back(
              Relation(RelationType::ResolvesBroadcast, idin, idout));
          continue;
        }
        id_partition_->mergeSetsFromValues(idin, idout);
        extent_partition_->mergeSetsFromValues(idin->extent(), idout->extent());
      }
    } else if (expr->isA<TransposeOp>()) {
      auto top = reinterpret_cast<TransposeOp*>(expr);
      auto inp = reinterpret_cast<TensorView*>(top->in());
      auto outp = reinterpret_cast<TensorView*>(top->out());
      auto indom = inp->domain()->domain();
      auto outdom = outp->domain()->domain();
      auto n2o = top->new2old();
      for (size_t i = 0; i < outdom.size(); ++i) {
        auto oldi = n2o[i];
        auto idin = indom[oldi];
        auto idout = outdom[i];
        id_partition_->mergeSetsFromValues(idin, idout);
        extent_partition_->mergeSetsFromValues(idin->extent(), idout->extent());
      }
    } else if (expr->isA<ViewOp>()) {
      std::cout << "Skipping reshape op: " << expr->toString() << std::endl;
      //} else if (expr->isA<ResizeOp>()) { // pending Naoya's recent work
      // TODO: Work through the list of other ops:
      /*
     *FullOp # nothing to do
     *ARangeOp # nothing to do
     *EyeOp # nothing to do
     *UnaryOp # handled by else
     *BinaryOp # handled by else
     *TernaryOp # handled by else
      SelectOp
      IndexSelectOp
      TorchGatherOp
      RNGOp
     *ReductionOp
      GroupedReductionOp
      WelfordOp
      GroupedWelfordOp
      LoadStoreOp
      MmaOp
     *BroadcastOp
      SqueezeOp
     *TransposeOp
      ExpandOp
      ShiftOp
      GatherOp
      ViewAsScalar
      ViewOp
      Split
      Merge
      Swizzle2D
      */
    } else {
      // For pointwise Exprs (or ExpandOp), this is most clear:
      //   - We simply merge matching (same position, ignoring reduction
      //   domains) serial IterDomain classes with one another.
      //   - If the input IterDomain is a bcast and the output Iterdomain is a
      //   bcast, we merge their e-classes.
      //   - If the input is a bcast and the output is serial (resolution of
      //   the broadcast), then we do not merge.
      //   Instead, in this case we add a relation that the e-class of the
      //   output ID _resolves_ the e-class of the input ID.
      for (auto inp_val : expr->inputs()) {
        if (inp_val->getValType() != ValType::TensorView) {
          continue;
        }
        auto inp = (TensorView*)inp_val;
        auto indom = inp->domain()->noReductions();
        for (auto outp_val : expr->outputs()) {
          if (outp_val->getValType() != ValType::TensorView) {
            continue;
          }
          auto outp = (TensorView*)outp_val;
          // Reduction domains aren't preserved through pointwise ops, so ignore
          // them For each non-reduction ID in inp and outp
          auto outdom = outp->domain()->noReductions();
          TORCH_CHECK(
              indom.size() == outdom.size(),
              "Input and output noReductions domains must have equal length in pointwise op");
          for (auto idin = indom.begin(), idout = outdom.begin();
               idin != indom.end() && idout != outdom.end();
               idin++, idout++) {
            TORCH_CHECK(
                !(*idout)->isBroadcast(),
                "Output IterDomains of pointwise ops should not be of broadcast type");
            if ((*idin)->isBroadcast()) {
              id_relations_.push_back(
                  Relation(RelationType::ResolvesBroadcast, *idin, *idout));
              continue;
            }
            id_partition_->mergeSetsFromValues(*idin, *idout);
            extent_partition_->mergeSetsFromValues(
                (*idin)->extent(), (*idout)->extent());
          }
        }
      }
    }
  }
  std::cout << "Equivalence classes of IterDomains:" << std::endl;
  for (auto s : id_partition_->getSets()) {
    std::cout << "  c";
    std::cout << id_partition_->findSetFromValue(s[0]) << ": ";
    for (auto id : s) {
      std::cout << id->toString() << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << "Equivalence classes of extents:" << std::endl;
  for (auto s : extent_partition_->getSets()) {
    std::cout << "  e";
    std::cout << extent_partition_->findSetFromValue(s[0]) << ": ";
    for (auto e : s) {
      std::cout << e->toString() << ", ";
    }
    std::cout << std::endl;
  }
}

//! Print out a diagram in GraphViz's .dot format
void IterDomainEGraph::printDot(std::ostream& stream) {
  stream << "digraph id_graph {" << std::endl;
  // print inputs
  stream << "  { rank = source;" << std::endl;
  // Each input tensor is printed with a partitioned box indicating the ID
  // name (not the class label), and edges are drawn from those to their
  // associated classes.
  for (auto in_val : fusion_.inputs()) {
    if (in_val->getValType() != ValType::TensorView) {
      continue;
    }
    auto tv = reinterpret_cast<TensorView*>(in_val);
    // Example line
    //   T0 [shape=record,label=""];
    stream << "    T" << tv->name() << " [shape=record, label=\"{T"
           << tv->name() << "|{";
    bool first = true;
    int i = 0;
    for (auto id : tv->domain()->domain()) {
      if (!first) {
        stream << "|"; // separator
      }
      stream << "<id" << i++ << "> " << id->getIterType()
             << id->getParallelType() << id->name();
      first = false;
    };
    stream << "}}\"];" << std::endl;
  }
  stream << "  }" << std::endl; // rank = min

  // Place edges from inputs to classes
  for (auto in_val : fusion_.inputs()) {
    if (in_val->getValType() != ValType::TensorView) {
      continue;
    }
    auto tv = reinterpret_cast<TensorView*>(in_val);
    auto i = 0;
    for (auto id : tv->domain()->domain()) {
      auto c = id_partition_->findSetFromValue(id);
      stream << "  T" << tv->name() << ":id" << i++ << " -> ";
      stream << "c" << c << ";" << std::endl;
    };
  }

  for (auto c : id_partition_.get()->getSetIndices()) {
    stream << "  c" << c << ";" << std::endl;
  }

  // Print all relations between classes
  for (auto r : id_relations_) {
    auto left_class = id_partition_->findSetFromValue(r.getLeft());
    auto right_class = id_partition_->findSetFromValue(r.getRight());
    // Place a labeled edge for every relation between ID classes
    stream << "  c" << left_class << " -> c" << right_class;
    stream << " [label=\"" << r.typeString() << "\"];" << std::endl;
  }

  // Place edges from classes to outputs
  for (auto out_val : fusion_.outputs()) {
    if (out_val->getValType() != ValType::TensorView) {
      continue;
    }
    auto tv = reinterpret_cast<TensorView*>(out_val);
    auto i = 0;
    for (auto id : tv->domain()->domain()) {
      auto c = id_partition_->findSetFromValue(id);
      stream << "  c" << c << " -> T";
      stream << tv->name() << ":id" << i++ << ";" << std::endl;
    };
  }

  // print outputs
  stream << "  { rank = max;" << std::endl;
  // Each output tensor is printed with a partitioned box indicating the ID
  // name (not the class label) ON TOP, and edges are drawn to those from
  // their associated classes.
  for (auto out_val : fusion_.outputs()) {
    if (out_val->getValType() != ValType::TensorView) {
      continue;
    }
    auto tv = reinterpret_cast<TensorView*>(out_val);
    // Example line
    //   T0 [shape=record,label=""];
    stream << "    T" << tv->name() << " [shape=record, label=\"{{";
    bool first = true;
    int i = 0;
    for (auto id : tv->domain()->domain()) {
      if (!first) {
        stream << "|"; // separator
      }
      stream << "<id" << i++ << "> " << id->getIterType()
             << id->getParallelType() << id->name();
      first = false;
    };
    stream << "}|T" << tv->name() << "}\"];" << std::endl;
  }
  stream << "  }" << std::endl; // rank = max

  // Add edges for relations

  stream << "}" << std::endl; // digraph id_graph {
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
