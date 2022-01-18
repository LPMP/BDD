#include "specialized_solvers/multi_graph_matching_input.h"
#include "specialized_solvers/graph_matching_input.h"
#include "specialized_solvers/graph_matching_torresani_et_al_grammar.h"
#include <tao/pegtl.hpp>
#include "pegtl_parse_rules.h"
#include <unordered_set>
#include "time_measure_util.h"

namespace LPMP {

    namespace Torresani_et_al_multi_graph_matching_parser {

        using parsing::mand_whitespace; 
        using parsing::opt_whitespace;  
        using parsing::positive_integer;

        struct comment_line : tao::pegtl::seq< opt_whitespace, tao::pegtl::sor<tao::pegtl::string<'c'>, tao::pegtl::string<'#'>>, tao::pegtl::until< tao::pegtl::eol >> {};
        struct empty_line : tao::pegtl::seq< opt_whitespace, tao::pegtl::eol > {};
        struct ignore_line : tao::pegtl::sor<comment_line, empty_line > {};

        struct graph_matching_line : tao::pegtl::seq< opt_whitespace, tao::pegtl::string<'g','m'>, mand_whitespace, positive_integer, mand_whitespace, positive_integer, opt_whitespace, tao::pegtl::eol > {};
        struct graph_matching : tao::pegtl::star<tao::pegtl::not_at<graph_matching_line>, tao::pegtl::any> {};

        struct grammar : tao::pegtl::seq<
                         tao::pegtl::star<
                         tao::pegtl::star<ignore_line>,
                         graph_matching_line,
                         graph_matching
                         >,
                         tao::pegtl::eof>
        {};

        template< typename Rule >
            struct action
            : tao::pegtl::nothing< Rule > {};

        template<> struct action< graph_matching_line > {
            template<typename INPUT>
                static void apply(const INPUT& in, multi_graph_matching_instance& instance, std::array<size_t,2>& cur_gm)
                {
                    std::istringstream iss(in.string());
                    char l;
                    iss >> l; assert(l == 'g');
                    iss >> l; assert(l == 'm');
                    size_t left_graph_nr; iss >> left_graph_nr;
                    size_t right_graph_nr; iss >> right_graph_nr;
                    assert(left_graph_nr < right_graph_nr);

                    cur_gm = {std::min(left_graph_nr, right_graph_nr), std::max(left_graph_nr, right_graph_nr)};
                }
        };

        template<> struct action< graph_matching > {
            template<typename INPUT>
                static void apply(const INPUT& in, multi_graph_matching_instance& instance, std::array<size_t,2>& cur_gm)
                {
                    graph_matching_instance gm_instance;
                    tao::pegtl::string_input input(in.string(), "graph matching input");
                    if(!tao::pegtl::parse<TorresaniEtAlInput::grammar, TorresaniEtAlInput::action>(input, gm_instance))
                        throw std::runtime_error("[multi-graph matching parser] Could not parse string");

                    instance.insert({cur_gm, gm_instance});
                }
        };
    }

    // check if all graph matchings i->j ,i<jare present
    bool is_full_mgm_instance(const multi_graph_matching_instance& mgm_instance)
    {
        size_t nr_graphs = 0; 
        for(const auto& [graph_nrs,gm_instance] : mgm_instance)
            nr_graphs = std::max({nr_graphs, graph_nrs[0]+1, graph_nrs[1]+1});

        std::unordered_set<std::array<size_t,2>> graph_matchings;
        for(const auto& [graph_nrs,gm_instance] : mgm_instance)
        {
            if(graph_nrs[0] > graph_nrs[1])
            {
                std::cout << "graph matching in reverse direction present\n";
                return false;
            }
            const size_t i = std::min(graph_nrs[0], graph_nrs[1]);
            const size_t j = std::max(graph_nrs[0], graph_nrs[1]);
            if(graph_matchings.count({i,j}) > 0)
            {
                std::cout << "duplicate graph matching present\n";
                return false;
            }
            graph_matchings.insert({i,j});
        }

        if(graph_matchings.size() != (nr_graphs * (nr_graphs-1))/2)
        {
            std::cout << "not all possible graph matchings present\n";
            return false;
        }

        return true;
    }

    ILP_input construct_multi_graph_matching_ILP(const multi_graph_matching_instance& mgm_instance)
    {
        assert(is_full_mgm_instance(mgm_instance));
    
        const auto& max_gm = std::get<0>(*std::max_element(mgm_instance.begin(), mgm_instance.end(), 
                [](const auto& a, const auto& b) {
                const size_t a_graph_nr_0 = std::get<0>(a)[0];
                const size_t a_graph_nr_1 = std::get<0>(a)[1];
                const size_t b_graph_nr_0 = std::get<0>(b)[0];
                const size_t b_graph_nr_1 = std::get<0>(b)[1];
                return std::max(a_graph_nr_0, a_graph_nr_1) < std::max(b_graph_nr_0, b_graph_nr_1);
                }));
        const size_t nr_graphs = std::max(max_gm[0], max_gm[1]) + 1;
        std::cout << "[construct multi-graph matching ILP]: #graphs = " << nr_graphs << "\n";

        std::unordered_map<std::array<size_t,2>, size_t> gm_var_offset;
        std::unordered_map<std::array<size_t,2>, std::unordered_map<std::array<size_t,2>,size_t>> linear_assignment_maps;
        ILP_input ilp;

        // add original graph matching constraints with offsets, record
        size_t var_offset = 0;
        for(const auto& [i,gm_instance] : mgm_instance)
        {
            assert(i[0] < i[1]);
            std::cout << "processing " << i[0] << "->" << i[1] << "\n";

            gm_var_offset.insert({{i[0],i[1]}, var_offset});

            auto [gm_ilp, linear_var_map, quadratic_var_map] = construct_graph_matching_ILP(gm_instance);
            for(size_t var=0; var<gm_ilp.nr_variables(); ++var)
            {
                const double obj_coeff = gm_ilp.objective(var);
                const size_t new_var = ilp.add_new_variable("mgm_" + std::to_string(i[0]) + "_" + std::to_string(i[1]) + "_" + gm_ilp.get_var_name(var));
                assert(new_var == var + var_offset);
                ilp.add_to_objective(obj_coeff, var + var_offset);
            }
            std::cout << "added objective\n";
            for(size_t c=0; c<gm_ilp.nr_constraints(); ++c)
            {
                auto constr = gm_ilp.constraints()[c];
                for(size_t c_idx=0; c_idx<constr.coefficients.size(); ++c_idx)
                    for(size_t m_idx=0; m_idx<constr.monomials.size(c_idx); ++m_idx)
                        constr.monomials(c_idx, m_idx) += var_offset;
                ilp.add_constraint(constr);
            }
            std::cout << "added constraints\n";

            // record linear assignment variables with offsets
            for(auto& [i_tmp, var] : linear_var_map)
                var += var_offset;
            //for(auto it=linear_var_map.begin(); it!=linear_var_map.end(); ++it)
            //    it.value() += var_offset;
            linear_assignment_maps.insert({i,linear_var_map});
            std::cout << "recorded linear vars\n";

            var_offset += gm_ilp.nr_variables();
        }
        std::cout << "[construct multi-graph matching ILP]: #graph matching constraints = " << ilp.constraints().size() << "\n";

        // add  cycle consistency constraints
        for(size_t i=0; i<nr_graphs; ++i)
        {
            for(size_t j=0; j<nr_graphs; ++j)
            {
                for(size_t k=0; k<nr_graphs; ++k)
                {
                    if(i != j && j != k && i < k)
                    {
                        struct assignment_elem { size_t ILP_var; size_t x_j; };

                        const bool ij_transposed = i>j;
                        const auto& ij_linear_vars = linear_assignment_maps.find({std::min(i,j), std::max(i,j)})->second;
                        std::vector<std::vector<assignment_elem>> ij_rows;
                        for(const auto [a_ij, ij_ILP_var] : ij_linear_vars)
                        {
                            const size_t x_i = ij_transposed ? a_ij[1] : a_ij[0];
                            const size_t x_j = ij_transposed ? a_ij[0] : a_ij[1];
                            if(x_i != graph_matching_instance::no_assignment && x_j != graph_matching_instance::no_assignment)
                            {
                                if(ij_rows.size() <= x_i)
                                    ij_rows.resize(x_i+1);
                                ij_rows[x_i].push_back({ij_ILP_var, x_j});
                            }
                        }

                        const bool jk_transposed = j>k;
                        const auto& jk_linear_vars = linear_assignment_maps.find({std::min(j,k), std::max(j,k)})->second;
                        std::vector<std::vector<assignment_elem>> jk_cols;
                        for(auto [a_jk, jk_ILP_var] : jk_linear_vars)
                        {
                            const size_t x_j = jk_transposed ? a_jk[1] : a_jk[0];
                            const size_t x_k = jk_transposed ? a_jk[0] : a_jk[1];
                            if(x_j != graph_matching_instance::no_assignment && x_k != graph_matching_instance::no_assignment)
                            {
                            if(jk_cols.size() <= x_k)
                                jk_cols.resize(x_k+1);
                            jk_cols[x_k].push_back({jk_ILP_var, x_j});
                            }
                        }

                        const auto& ik_linear_vars = linear_assignment_maps.find({i,k})->second;

                        // X_ij * X_jk <= X_ik
                        for(const auto [a_ik,ILP_var] : ik_linear_vars)
                        {
                            const size_t x_i = a_ik[0];
                            const size_t x_k = a_ik[1];
                            if(x_i != graph_matching_instance::no_assignment && x_k != graph_matching_instance::no_assignment)
                            {
                                bool monomials_found = false;
                                size_t ij_counter = 0;
                                size_t jk_counter = 0;
                                for(; x_i < ij_rows.size() && x_k < jk_cols.size() && ij_counter<ij_rows[x_i].size() && jk_counter<jk_cols[x_k].size();)
                                {
                                    if(ij_rows[x_i][ij_counter].x_j == jk_cols[x_k][jk_counter].x_j)
                                    {
                                        std::array<size_t,2> monomial = {ij_rows[x_i][ij_counter].ILP_var, jk_cols[x_k][jk_counter].ILP_var};
                                        if(monomials_found == false)
                                        {
                                            monomials_found = true;
                                            ilp.begin_new_inequality();
                                        }
                                        ilp.add_to_constraint(1, monomial.begin(), monomial.end());
                                        ++ij_counter;
                                        ++jk_counter;
                                    }
                                    else if(ij_rows[x_i][ij_counter].x_j < jk_cols[x_k][jk_counter].x_j)
                                    {
                                        ++ij_counter;
                                    }
                                    else
                                    {
                                        assert(ij_rows[x_i][ij_counter].x_j > jk_cols[x_k][jk_counter].x_j);
                                        ++jk_counter;
                                    }
                                }
                                if(monomials_found)
                                {
                                    ilp.add_to_constraint(-1, ILP_var);
                                    ilp.set_inequality_type(ILP_input::inequality_type::smaller_equal);
                                    ilp.set_right_hand_side(0);
                                }
                            }
                        }
                    }
                }
            }
        }

        return ilp;
    }

    ILP_input parse_multi_graph_matching_file(const std::string& filename)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        std::array<size_t,2> cur_gm;
        multi_graph_matching_instance mgm_instance;
        std::cout << "[multi-graph matching parser] Parse multi-graph matching file " << filename << "\n";

        tao::pegtl::file_input input(filename);
        if(!tao::pegtl::parse<Torresani_et_al_multi_graph_matching_parser::grammar, Torresani_et_al_multi_graph_matching_parser::action>(input, mgm_instance, cur_gm))
            throw std::runtime_error("[multi-graph matching parser] Could not read file" + filename);

        for(const auto& [i, gm_instance] : mgm_instance)
            std::cout << "[multi-graph matching parser] graph matching problem " << i[0] << " -> " << i[1] << " has " << gm_instance.linear_assignments.size() << " linear and " << gm_instance.quadratic_assignments.size() << " quadratic assignments\n";

        return construct_multi_graph_matching_ILP(mgm_instance);
    }

    ILP_input parse_multi_graph_matching_string(const std::string& mgm_string)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        std::array<size_t,2> cur_gm;
        multi_graph_matching_instance mgm_instance;
        std::cout << "[multi-graph matching parser] Parse multi-graph matching string\n";

        tao::pegtl::string_input input(mgm_string, "multi-graph matching input");
        if(!tao::pegtl::parse<Torresani_et_al_multi_graph_matching_parser::grammar, Torresani_et_al_multi_graph_matching_parser::action>(input, mgm_instance, cur_gm))
            throw std::runtime_error("[multi-graph matching parser] Could not read string");

        return construct_multi_graph_matching_ILP(mgm_instance);
    }
}
