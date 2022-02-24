#pragma once

#include "bdd_manager/bdd.h"
#include "bdd_collection/bdd_collection.h"
#include "ILP_input.h"
#include "convert_pb_to_bdd.h"
#include "lineq_bdd.h"
#include "two_dimensional_variable_array.hxx"
#include <cassert>
#include <vector>

namespace LPMP {
    
    class bdd_preprocessor {
        public:
            bdd_preprocessor() {};
            bdd_preprocessor(const ILP_input& ilp, const bool constraint_groups = true, const bool normalize = false)
            {
                add_ilp(ilp, constraint_groups, normalize);
            }

            two_dim_variable_array<size_t> add_ilp(const ILP_input& ilp, const bool constraint_groups = true, const bool normalize = false);
            template<typename VARIABLE_ITERATOR>
                void add_bdd(BDD::node_ref bdd, VARIABLE_ITERATOR var_begin, VARIABLE_ITERATOR var_end);

            void add_bdd(BDD::node_ref bdd);
            void add_bdd(BDD::bdd_collection_entry bdd);

            size_t nr_bdds() const { return bdd_collection.nr_bdds(); }

            BDD::bdd_collection& get_bdd_collection() { return bdd_collection; }

            void set_coalesce_bridge() { coalesce_bridge_ = true; }
            void set_coalesce_subsumption() { coalesce_subsumption_ = true; }
            void set_coalesce_contiguous_overlap() { coalesce_contiguous_overlap_ = true; }
            void set_coalesce_subsumption_except_one() { coalesce_subsumption_except_one_ = true; }
            void set_coalesce_partial_contiguous_overlap() { coalesce_partial_contiguous_overlap_ = true; }
            void set_coalesce_cliques() { coalesce_cliques_ = true; }

            void coalesce();
            void coalesce_bdd_collection();
            //BDD::bdd_mgr& get_bdd_manager() { return bdd_mgr; }
            //std::vector<BDD::node_ref> get_bdds() { return bdds; }

            // obvious candidates for coalescing
            //enum class coalesce_candidate_type {
            //    bridge = 0,
            //    subsumption = 1,
            //    contiguous_overlap = 2,
            //    subsumption_except_one = 3,
            //    partial_contiguous_overlap = 4
            //};
            struct coalesce_candidate : public std::array<size_t,2> {
            //    coalesce_candidate_type type;
            //    bool operator<(const coalesce_candidate& o) { return type < o.type; }
                bool operator<(const coalesce_candidate& o) const { 
                    if((*this)[0] == o[0]) return (*this)[0] < o[0];
                    return (*this)[1] < o[1];
                }
            };
            std::vector<coalesce_candidate> bridge_candidates(const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const;
            std::vector<coalesce_candidate> subsumption_candidates(const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const { return compute_candidates(true, false, false, false, bdd_var_adjacency, var_bdd_adjacency); }
            std::vector<coalesce_candidate> contiguous_overlap_candidates(const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const { return compute_candidates(false, true, false, false, bdd_var_adjacency, var_bdd_adjacency); }
            std::vector<coalesce_candidate> subsumption_except_one_candidates(const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const { return compute_candidates(false, true, false, false, bdd_var_adjacency, var_bdd_adjacency); }
            std::vector<coalesce_candidate> partial_contiguous_overlap_candidates(const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const { return compute_candidates(false, false, false, true, bdd_var_adjacency, var_bdd_adjacency); }

            std::vector<coalesce_candidate> compute_candidates(const bool subsumption, const bool contiguous_overlap, const bool subsumption_except_one, const bool partial_contiguous_overlap, const two_dim_variable_array<size_t>& bdd_var_adjacency, const two_dim_variable_array<size_t>& var_bdd_adjacency) const;

            //void coalesce_cliques();
            //struct empty{};
            //using adjacency_graph = graph<empty>;
        private:

            // return {bdd_var_adjacency, var_bdd_adjacency};
            template<typename BDDS>
                std::tuple<two_dim_variable_array<size_t>,two_dim_variable_array<size_t>> construct_bdd_var_adjacency(BDDS& bdds) const;

            //std::tuple<bdd_preprocessor::adjacency_graph, two_dim_variable_array<size_t>, two_dim_variable_array<size_t>>
            //compute_var_adjacency(const size_t bdd_size_limit = std::numeric_limits<size_t>::max());

            BDD::bdd_mgr bdd_mgr;
            BDD::bdd_collection bdd_collection;
            std::vector<BDD::node_ref> bdds;
            size_t nr_variables = 0;

            bool coalesce_bridge_ = false;
            bool coalesce_subsumption_ = false;
            bool coalesce_contiguous_overlap_ = false;
            bool coalesce_subsumption_except_one_ = false;
            bool coalesce_partial_contiguous_overlap_ = false;
            bool coalesce_cliques_ = false;
    };

    template<typename VARIABLE_ITERATOR>
        void bdd_preprocessor::add_bdd(BDD::node_ref bdd, VARIABLE_ITERATOR var_begin, VARIABLE_ITERATOR var_end)
        {
            bdds.push_back(bdd);
            nr_variables = std::max(nr_variables, *(var_end-1)+1);
        }

    template<typename BDDS>
    std::tuple<two_dim_variable_array<size_t>,two_dim_variable_array<size_t>> bdd_preprocessor::construct_bdd_var_adjacency(BDDS& bdds) const
    {
        two_dim_variable_array<size_t> bdd_var_adjacency;

        std::vector<size_t> var_bdd_adjacency_size(nr_variables, 0);
        // first compute variable bdd incidence matrix
        for(size_t j=0; j<bdds.size(); ++j)
        {
            const auto vars = bdds[j].variables();
            bdd_var_adjacency.push_back(vars.begin(), vars.end());
            for(const size_t v : vars)
            {
                assert(v < var_bdd_adjacency_size.size());
                var_bdd_adjacency_size[v]++;
            }
        }

        two_dim_variable_array<size_t> var_bdd_adjacency(var_bdd_adjacency_size.begin(), var_bdd_adjacency_size.end());
        std::fill(var_bdd_adjacency_size.begin(), var_bdd_adjacency_size.end(), 0);
        for(size_t j=0; j<bdd_var_adjacency.size(); ++j)
            for(const size_t v : bdd_var_adjacency[j])
                var_bdd_adjacency(v, var_bdd_adjacency_size[v]++) = j;

        return {bdd_var_adjacency, var_bdd_adjacency};
    }

}
