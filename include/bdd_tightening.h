#pragma once

#include "bdd_manager/bdd.h"
#include "bdd_collection/bdd_collection.h"
#include "union_find.hxx"
#include "min_marginal_utils.h"
#include <unordered_set>
#include <fstream> // TODO: remove

namespace LPMP {

    // partition BDDs into several groups such that they are disconnected w.r.t. variables covered.
    // returns indices and bdd numbers that make up groups
    template<typename ITERATOR>
        std::tuple<two_dim_variable_array<size_t>,two_dim_variable_array<size_t>> bdd_groups(BDD::bdd_collection& bdd_col, ITERATOR bdd_begin, ITERATOR bdd_end)
        {
            // find connected components of these BDDs.
            two_dim_variable_array<size_t> vars;
            size_t nr_vars = 0;
            for(auto bdd_nr_it=bdd_begin; bdd_nr_it!=bdd_end; ++bdd_nr_it)
            {
                const size_t bdd_nr = *bdd_nr_it;
                const auto bdd_vars = bdd_col.variables(bdd_nr);
                vars.push_back(bdd_vars.begin(), bdd_vars.end());
                nr_vars = std::max(nr_vars, *std::max_element(bdd_vars.begin(), bdd_vars.end())+1);
            }

            union_find bdd_cc_uf(nr_vars);
            for(size_t g=0; g<vars.size(); ++g)
            {
                for(size_t v_idx=1; v_idx<vars.size(g); ++v_idx)
                {
                    const size_t prev_var = vars(g,v_idx-1);
                    const size_t cur_var = vars(g,v_idx);
                    bdd_cc_uf.merge(prev_var, cur_var); 
                }
            }

            std::unordered_map<size_t, std::vector<size_t>> bdd_cc;
            for(size_t bdd_idx=0; bdd_idx<vars.size(); ++bdd_idx)
            {
                assert(vars.size(bdd_idx) > 0);
                const size_t cc_id = bdd_cc_uf.find(vars(bdd_idx, 0));
                if(bdd_cc.count(cc_id) == 0)
                    bdd_cc.insert({cc_id, {}});
                std::vector<size_t>& cc = bdd_cc.find(cc_id)->second;
                cc.push_back(bdd_idx); 
            }

            two_dim_variable_array<size_t> bdd_nrs_groups;
            two_dim_variable_array<size_t> idx_groups;
            for(const auto [cc_id, idx] : bdd_cc)
            {
                idx_groups.push_back(idx.begin(), idx.end());
                std::vector<size_t> bdd_nrs(idx.begin(), idx.end());
                for(size_t& i : bdd_nrs)
                {
                    assert(i < std::distance(bdd_begin, bdd_end));
                    i = *(bdd_begin + i); 
                }
                bdd_nrs_groups.push_back(bdd_nrs.begin(), bdd_nrs.end());
            }

            std::cout << "# connected components for BDD intersection: " << bdd_cc.size() << "\n";
            std::cout << "# bdds per cc: ";
            for(size_t g=0; g<bdd_nrs_groups.size(); ++g)
                std::cout << bdd_nrs_groups.size(g) << ", ";
            std::cout << "\n";

            return {idx_groups, bdd_nrs_groups};
        }

    enum class variable_fixation {
        zero,
        one,
        not_set 
    };

    std::vector<variable_fixation> variable_fixations(const std::vector<float>& min_marg_diffs, const float eps)
    {
        // compute variables to relax
        std::vector<variable_fixation> var_fixes;
        var_fixes.reserve(min_marg_diffs.size());
        for(size_t v=0; v<min_marg_diffs.size(); ++v)
        {
            if(min_marg_diffs[v] > eps)
                var_fixes.push_back(variable_fixation::zero);
            else if(min_marg_diffs[v] < -eps)
                var_fixes.push_back(variable_fixation::one);
            else
                var_fixes.push_back(variable_fixation::not_set); 
        }

        std::cout << "zero min marg diff vars:\n";
        for(size_t i=0; i<min_marg_diffs.size(); ++i)
        {
            if(var_fixes[i] == variable_fixation::not_set)
                std::cout << " " << i;
        }
        std::cout << "\n";

        return var_fixes; 
    }

    template<typename BDD_SOLVER>
        void tighten_per_variable(BDD_SOLVER& s, float eps)
        {
            assert(eps > 0.0);
            BDD::bdd_collection bdd_col;
            BDD::bdd_mgr bdd_mgr;

            const two_dim_variable_array<std::array<float,2>> min_marginals = s.min_marginals();
            const std::vector<float> min_marg_diffs = min_marginal_differences(min_marginals, eps);
            const std::vector<variable_fixation> fixed_vars = variable_fixations(min_marg_diffs, eps);

            // compute intersection of all covering BDDs for each variable

        }

    // export BDDs covering any variable with zero min marginal difference
    // return bdd collection, BDD numbers for bdd collection, BDD numbers in solver, its corresponding variables
    template<typename BDD_SOLVER>
        std::tuple<BDD::bdd_collection, std::vector<size_t>, two_dim_variable_array<size_t>, std::vector<size_t>>
        bdd_groups(BDD_SOLVER& s, float eps)
        {
            assert(eps > 0.0);
            BDD::bdd_collection bdd_col;
            BDD::bdd_mgr bdd_mgr;

            // first get min-marginals for each BDD and each variable.
            const two_dim_variable_array<std::array<float,2>> min_marginals = s.min_marginals();
            const std::vector<float> min_marg_diffs = min_marginal_differences(min_marginals, eps);
            const std::vector<variable_fixation> fixed_vars = variable_fixations(min_marg_diffs, eps);

            //std::cout << "# positive vars = " << positive_vars.size() << ", # negative vars = " << negative_vars.size() << ", # remaining vars = " << s.nr_variables() - positive_vars.size() - negative_vars.size() << "\n";

            // get all BDDs that cover at least one variable with small min-marginal difference. Weaken them with the other variables.
            std::vector<size_t> solver_bdd_nrs; // affected BDDs from solver
            std::vector<BDD::node_ref> bdds; // exported and converted BDDs from solver
            std::vector<size_t> bdd_col_nrs; // bdd numbers from bdd_collection
            two_dim_variable_array<size_t> bdd_variables;
            std::cout << "# bdds in solver = " << s.nr_bdds() << "\n";
            for(size_t bdd_nr=0; bdd_nr<s.nr_bdds(); ++bdd_nr)
            {
                const auto vars = s.variables(bdd_nr);
                for(const size_t v : vars)
                {
                    if(fixed_vars[v] == variable_fixation::not_set)
                    {
                        solver_bdd_nrs.push_back(bdd_nr);

                        auto [bdd, bdd_vars] = s.export_bdd(bdd_mgr, bdd_nr);
                        bdds.push_back(bdd);
                        bdd_col_nrs.push_back(bdd_col.add_bdd(bdds.back()));
                        bdd_col.rebase(bdd_col_nrs.back(), bdd_vars.begin(), bdd_vars.end());

                        bdd_variables.push_back(vars.begin(), vars.end());
                        break;
                    }
                }
            }
            //std::cout << "# bdds to investigate for tightening = " << bdd_nrs.size() << "\n";
            std::cout << "# bdds for tightening = " << bdd_col_nrs.size() << "\n";

            return {bdd_col, bdd_col_nrs, bdd_variables, solver_bdd_nrs};
        }

    template<typename BDD_SOLVER>
        void tighten(BDD_SOLVER& s, float eps)
        {
            assert(eps > 0.0);
            eps = 1e-6;
            BDD::bdd_collection bdd_col;
            BDD::bdd_mgr bdd_mgr;

            // first get min-marginals for each BDD and each variable.
            const two_dim_variable_array<std::array<float,2>> min_marginals = s.min_marginals();
            const std::vector<float> min_marg_diffs = min_marginal_differences(min_marginals, eps);
            const std::vector<variable_fixation> fixed_vars = variable_fixations(min_marg_diffs, eps);

            //std::cout << "# positive vars = " << positive_vars.size() << ", # negative vars = " << negative_vars.size() << ", # remaining vars = " << s.nr_variables() - positive_vars.size() - negative_vars.size() << "\n";

            // get all BDDs that cover at least one variable with small min-marginal difference. Weaken them with the other variables.
            std::vector<size_t> solver_bdd_nrs; // affected BDDs from solver
            std::vector<BDD::node_ref> bdds; // exported and converted BDDs from solver
            std::vector<size_t> bdd_col_nrs; // bdd numbers from bdd_collection
            two_dim_variable_array<float> bdd_costs;
            two_dim_variable_array<size_t> bdd_variables;
            std::cout << "# bdds in solver = " << s.nr_bdds() << "\n";
            for(size_t bdd_nr=0; bdd_nr<s.nr_bdds(); ++bdd_nr)
            {
                const auto vars = s.variables(bdd_nr);
                for(const size_t v : vars)
                {
                    if(std::abs(min_marg_diffs[v]) <= eps)
                    {
                        solver_bdd_nrs.push_back(bdd_nr);

                        auto [bdd, bdd_vars] = s.export_bdd(bdd_mgr, bdd_nr);
                        bdds.push_back(bdd);
                        bdd_col_nrs.push_back(bdd_col.add_bdd(bdds.back()));
                        bdd_col.rebase(bdd_col_nrs.back(), bdd_vars.begin(), bdd_vars.end());
                        std::unordered_set<size_t> positive_vars;
                        std::unordered_set<size_t> negative_vars;
                        for(const auto i : bdd_vars)
                        {
                            if(fixed_vars[i] == variable_fixation::one)
                                positive_vars.insert(i);
                            else if(fixed_vars[i] == variable_fixation::zero)
                                negative_vars.insert(i);
                        }

                        bdd_col_nrs.back() = bdd_col.bdd_or_var(bdd_col_nrs.back(), positive_vars, negative_vars);
                        if(bdd_col.nr_bdd_nodes(bdd_col_nrs.back()) <= 2)
                        {
                            bdds.resize(bdds.size()-1);
                            bdd_col_nrs.resize(bdd_col_nrs.size()-1);
                            break;
                        }

                        const auto costs = s.get_costs(bdd_nr);
                        bdd_costs.push_back(costs.begin(), costs.end()); 

                        bdd_variables.push_back(vars.begin(), vars.end());
                        break;
                    }
                }
            }
            //std::cout << "# bdds to investigate for tightening = " << bdd_nrs.size() << "\n";
            std::cout << "# bdds used in tightening = " << bdds.size() << "\n";

            const auto [bdd_group_idx, bdd_group_nrs] = bdd_groups(bdd_col, bdd_col_nrs.begin(), bdd_col_nrs.end());

            //assert(bdd_group_idx.size() <= s.nr_variables() - positive_vars.size() - negative_vars.size());

            std::vector<size_t> intersect_bdd_nrs;
            for(size_t g=0; g<bdd_group_nrs.size(); ++g)
            {
                std::cout << "intersected bdd group " << g;
                std::cout << ", has " << bdd_group_nrs.size(g) << " bdds";
                std::cout << std::flush;
                intersect_bdd_nrs.push_back(bdd_col.bdd_and(bdd_group_nrs[g].begin(), bdd_group_nrs[g].end()));
                std::cout << ", resulting bdd has " << bdd_col.nr_bdd_nodes(intersect_bdd_nrs.back()) << " nodes\n";
                intersect_bdd_nrs.back() = bdd_col.make_qbdd(intersect_bdd_nrs.back()); 
                assert(bdd_col.fixed_variables(intersect_bdd_nrs.back())[0].size() == 0);
                assert(bdd_col.fixed_variables(intersect_bdd_nrs.back())[1].size() == 0);
            }

            const auto new_solver_bdd_nrs = s.add_bdds(bdd_col, intersect_bdd_nrs.begin(), intersect_bdd_nrs.end());
            //assert(new_solver_bdd_nrs.size() == 1);

            // put costs of affected BDDs into costs of new BDD
            for(size_t g=0; g<bdd_group_idx.size(); ++g)
            {
                const size_t solver_intersect_bdd_nr = new_solver_bdd_nrs[g];
                for(size_t idx=0; idx<bdd_group_idx.size(g); ++idx)
                {
                    const size_t solver_bdd_nr = solver_bdd_nrs[bdd_group_idx(g,idx)];
                    auto costs = s.get_costs(solver_bdd_nr);
                    const auto vars = s.variables(solver_bdd_nr);
                    s.update_bdd_costs(solver_intersect_bdd_nr,
                            costs.begin(), costs.end(),
                            vars.begin(), vars.end());

                    for(auto& x : costs)
                        x *= -1.0;
                    s.update_bdd_costs(solver_bdd_nr,
                            costs.begin(), costs.end(),
                            vars.begin(), vars.end()); 
                } 
            }
        }

}
