#pragma once

#include "bdd_branch_instruction.h"
#include "bdd.h"
#include "union_find.hxx"

namespace LPMP {

    // TODO: templatize w.r.t. floating point

    template<typename REAL>
    std::vector<REAL> min_marginal_differences(const two_dim_variable_array<std::array<REAL,2>>& min_marginals, const REAL eps)
    {
        std::vector<REAL> mmd;
        mmd.reserve(min_marginals.size());
        for(size_t var=0; var<min_marginals.size(); ++var)
        {
            const size_t nr_bdds = min_marginals.size(var);
            REAL diff_sum = 0.0; //std::numeric_limits<REAL>::infinity(); 
            bool negative = true;
            bool positive = true;
            for(size_t i=0; i<nr_bdds; ++i)
            {
                diff_sum += std::abs(min_marginals(var,i)[1] - min_marginals(var,i)[0]);
                assert(std::isfinite(min_marginals(var,i)[0]));
                assert(std::isfinite(min_marginals(var,i)[1]));
                if(min_marginals(var,i)[1] - min_marginals(var,i)[0] >= eps)
                {
                    negative = false; 
                }
                else if(min_marginals(var,i)[1] - min_marginals(var,i)[0] <= -eps)
                {
                    positive = false;
                }
                else
                {
                    negative = false;
                    positive = false; 
                }
            }
            assert(nr_bdds == 0 || (!(positive == true && negative == true)));
            if(negative)
                mmd.push_back(-diff_sum);
            else if(positive)
                mmd.push_back(diff_sum);
            else
                mmd.push_back(0.0);
        }

        const size_t nr_positive_min_marg_differences = std::count_if(mmd.begin(), mmd.end(), [&](const float x) { return x > eps; });
        const size_t nr_negative_min_marg_differences = std::count_if(mmd.begin(), mmd.end(), [&](const float x) { return x < -eps; });
        const size_t nr_zero_min_marg_differences = mmd.size() - nr_positive_min_marg_differences - nr_negative_min_marg_differences;
        std::cout << "%zero min margs = " << 100.0 * double(nr_zero_min_marg_differences) / double(mmd.size()) << "\n";
        std::cout << "#zero min margs = " << nr_zero_min_marg_differences << "\n";
        std::cout << "%positive min margs = " << 100.0 * double(nr_positive_min_marg_differences) / double(mmd.size()) << "\n";
        std::cout << "#positive min margs = " << nr_positive_min_marg_differences << "\n";
        std::cout << "%negative min margs = " << 100.0 * double(nr_negative_min_marg_differences) / double(mmd.size()) << "\n";
        std::cout << "#negative min margs = " << nr_negative_min_marg_differences << "\n";

        return mmd;
    }

    template<typename BDD_SOLVER>
        void tighten(BDD_SOLVER& s, const float eps)
        {
            BDD::bdd_collection bdd_col;
            BDD::bdd_mgr bdd_mgr;
            std::vector<BDD::node_ref> node_refs;

            // first get min-marginals for each BDD and each variable.
            const two_dim_variable_array<std::array<float,2>> min_marginals = s.min_marginals();
            const std::vector<float> min_marg_diffs = min_marginal_differences(min_marginals, eps);
            std::cout << "min marg diffs: ";
            for(const float d : min_marg_diffs)
                std::cout << d << ", ";
            std::cout << "\n";
           
            // get those variables that have zero min-marginal difference or contradicting min-marginal differences.

            // get all those BDDs that cover at least one one found variables. Weaken them with the rest of the variables.
            std::vector<size_t> bdd_nrs;
            two_dim_variable_array<size_t> bdd_variables;
            std::cout << "# bdds in solver = " << s.nr_bdds() << "\n";
            for(size_t bdd_nr=0; bdd_nr<s.nr_bdds(); ++bdd_nr)
            {
                const auto vars = s.variables(bdd_nr);
                for(const size_t v : vars)
                {
                    if(min_marg_diffs[v] <= eps)
                    {
                        bdd_nrs.push_back(s.export_bdd(bdd_col, bdd_nr));
                        node_refs.push_back(bdd_mgr.add_bdd(bdd_col, bdd_nrs.back()));
                        // TODO: perform or_var to relax variables with good min marginals
                        bdd_variables.push_back(vars.begin(), vars.end());
                        break;
                    }
                }
            }
            std::cout << "# bdds to investigate for tightening = " << bdd_nrs.size() << "\n";

            // find connected components of these BDDs.
            union_find bdd_cc_uf(s.nr_variables());
            for(size_t bdd_idx=0; bdd_idx<bdd_variables.size(); ++bdd_idx)
            {
                for(size_t v_idx=1; v_idx<bdd_variables.size(bdd_idx); ++v_idx)
                {
                    const size_t prev_var = bdd_variables(bdd_idx, v_idx-1);
                    const size_t cur_var = bdd_variables(bdd_idx, v_idx);
                    bdd_cc_uf.merge(prev_var, cur_var); 
                } 
            }

            std::unordered_map<size_t, std::vector<size_t>> bdd_cc;
            for(size_t bdd_idx=0; bdd_idx<bdd_variables.size(); ++bdd_idx)
            {
                assert(bdd_variables.size(bdd_idx) > 0);
                const size_t cc_id = bdd_cc_uf.find(bdd_variables(bdd_idx, 0));
                if(bdd_cc.count(cc_id) == 0)
                    bdd_cc.insert({cc_id, {}});
                std::vector<size_t>& cc = bdd_cc.find(cc_id)->second;
                cc.push_back(bdd_idx); 
            }

            std::cout << "# connected components for BDD intersection: " << bdd_cc.size() << "\n";

            BDD::node_ref bdd_intersect = bdd_mgr.and_rec(node_refs.begin(), node_refs.end());
            const size_t new_bdd_nr = bdd_col.add_bdd(bdd_intersect);

            // intersect
            //std::vector<size_t> new_bdd_nrs;
            //for(auto [cc_id, bdd_vec] : bdd_cc)
            //    new_bdd_nrs.push_back(bdd_col.bdd_and(bdd_vec.begin(), bdd_vec.end()));

            // add new bdds to solver
            //bdd_storage stor;
            //for(const size_t new_bdd_nr : new_bdd_nrs)
            //    stor.add_bdd(bdd_col[new_bdd_nr]);
            //s.add_bdds(stor);
        }


}
