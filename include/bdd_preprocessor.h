#pragma once

#include "bdd_collection/bdd_collection.h"
#include "ILP_input.h"
#include "convert_pb_to_bdd.h"
#include "two_dimensional_variable_array.hxx"
#include <cassert>
#include <vector>

namespace LPMP {
    
    class bdd_preprocessor {
        public:
            bdd_preprocessor() {};
            bdd_preprocessor(const ILP_input& ilp, const bool normalize = false, const bool split_long_bdds = false, const bool add_split_implication_bdd = false)
            {
                add_ilp(ilp, normalize, split_long_bdds, add_split_implication_bdd);
            }

            two_dim_variable_array<size_t> add_ilp(const ILP_input& ilp, const bool normalize = false, const bool split_long_bdds = false, const bool add_split_implication_bdd = false);
            template<typename VARIABLE_ITERATOR>
                void add_bdd(BDD::node_ref bdd, VARIABLE_ITERATOR var_begin, VARIABLE_ITERATOR var_end);

            void add_bdd(BDD::node_ref bdd);
            void add_bdd(BDD::bdd_collection_entry bdd);

            size_t nr_bdds() const { return bdd_collection.nr_bdds(); }

            BDD::bdd_collection& get_bdd_collection() { return bdd_collection; }

        private:

            BDD::bdd_collection bdd_collection;
            size_t nr_variables = 0;

    };

}
