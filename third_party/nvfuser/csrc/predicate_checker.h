#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <expr_evaluator.h>
#include <ir_base_nodes.h>

#include <iostream>
#include <string>
#include <vector>

namespace nvfuser {

//! This class holds a collection of Bool objects, which are boolean-valued
//! Scalars. These may be constants or they may be derived via Exprs.
class PredicateChecker {
 public:
  void insert(const Bool* p) {
    predicates_.push_back(p);
  }

  //! Evaluate each predicate and return a vector of their values
  std::vector<bool> each(std::vector<const Val*> inputs);

  //! Boolean ALL of all registered predicates, evaluated with provided inputs
  bool all(std::vector<const Val*> inputs);

  //! Boolean ALL of all registered predicates, evaluated with current fusion
  //! inputs
  bool all();

  //! Boolean OR of all registered predicates, evaluated with provided inputs
  bool any(std::vector<const Val*> inputs);

  //! Boolean OR of all registered predicates, evaluated with current fusion
  //! inputs
  bool any();

  std::string toString() {
    std::stringstream ss;
    ss << "PredicateChecker {";
    bool first = true;
    for (auto p : predicates_) {
      if (first) {
        ss << std::endl;
        first = false;
      }
      ss << "  " << p << std::endl;
    }
    ss << "}";
    return ss.str();
  }

 private:
  std::vector<const Bool*> predicates_;
  ExpressionEvaluator evaluator_;
};

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    PredicateChecker& pc) {
  os << pc.toString();
  return os;
}

} // namespace nvfuser
