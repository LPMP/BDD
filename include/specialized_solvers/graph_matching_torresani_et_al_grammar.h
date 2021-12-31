#pragma once
#include <tao/pegtl.hpp>
#include "pegtl_parse_rules.h"

namespace LPMP {

    // grammar for reading in files in the format of the Dual Decomposition algorithm of Torresani, Kolmogorov and Rother
    namespace TorresaniEtAlInput {

        /* file format              
        // Angular parentheses mean that it should be replaced with an integer number,
        // curly parentheses mean a floating point number.
        // Point and assignment id's are integers starting from 0.

        c comment line
        p <N0> <N1> <A> <E>     // # points in the left image, # points in the right image, # assignments, # edges
        a <a> <i0> <i1> {cost}  // specify assignment
        e <a> <b> {cost}        // specify edge

        i0 <id> {xi} {yi}       // optional - specify coordinate of a point in the left image
        i1 <id> {xi} {yi}       // optional - specify coordinate of a point in the left image
        n0 <i> <j>              // optional - specify that points <i> and <j> in the left image are neighbors
        n1 <i> <j>              // optional - specify that points <i> and <j> in the right image are neighbors
        */ 

        using parsing::mand_whitespace; 
        using parsing::opt_whitespace;  
        using parsing::positive_integer;
        using parsing::real_number;


        // first two integers are number of left nodes, number of right nodes, then comes number of assignments, and then number of quadratic terms
        struct no_left_nodes : tao::pegtl::seq< positive_integer > {};
        struct no_right_nodes : tao::pegtl::seq< positive_integer > {};
        struct init_line : tao::pegtl::seq< opt_whitespace, tao::pegtl::string<'p'>, mand_whitespace, no_left_nodes, mand_whitespace, no_right_nodes, mand_whitespace, positive_integer, mand_whitespace, positive_integer, opt_whitespace > {};
        // numbers mean: assignment number (consecutive), then comes left node number, right node number, cost
        struct assignment : tao::pegtl::seq < positive_integer, mand_whitespace, positive_integer, mand_whitespace, positive_integer, mand_whitespace, real_number > {};
        struct assignmentsline : tao::pegtl::seq< opt_whitespace, tao::pegtl::string<'a'>, mand_whitespace, assignment, opt_whitespace> {}; 
        // numbers mean: number of left assignment, number of right assignment, cost
        struct quadratic_pot : tao::pegtl::seq< positive_integer, mand_whitespace, positive_integer, mand_whitespace, real_number > {};
        struct quadratic_pot_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'e'>, mand_whitespace, quadratic_pot, opt_whitespace > {};

        struct comment_line : tao::pegtl::seq< opt_whitespace, tao::pegtl::string<'c'>, tao::pegtl::until< tao::pegtl::eolf >> {}; 

        // artifacts from dual decomposition file format. We do not make use of them.
        struct neighbor_line : tao::pegtl::seq< tao::pegtl::sor<tao::pegtl::string<'n','0'>, tao::pegtl::string<'n','1'>>, tao::pegtl::until< tao::pegtl::eolf>> {};
        struct coordinate_line : tao::pegtl::seq< tao::pegtl::sor<tao::pegtl::string<'i','0'>, tao::pegtl::string<'i','1'>>, tao::pegtl::until< tao::pegtl::eolf>> {};

        // better way to cope with comment lines? On each line there may be a comment
        struct grammar : tao::pegtl::must<
                            tao::pegtl::star<comment_line>,
                            init_line,tao::pegtl::eol,
                            tao::pegtl::star<
                                 tao::pegtl::sor<
                                    tao::pegtl::seq<quadratic_pot_line, tao::pegtl::eol>,
                                    tao::pegtl::seq<assignmentsline, tao::pegtl::eol>,
                                    comment_line,
                                    neighbor_line,
                                    coordinate_line,
                                    tao::pegtl::seq<opt_whitespace, tao::pegtl::eol>
                                >
                            >
                            //, tao::pegtl::eof
                         > {};

        template< typename Rule >
            struct action
            : tao::pegtl::nothing< Rule > {};    

        template<> struct action< no_left_nodes > {
            template<typename INPUT>
                static void apply(const INPUT& in, graph_matching_instance& gmInput)
                {
                    // gmInput.no_left_nodes = std::stoul(in.string());
                }
        };

        template<> struct action< no_right_nodes > {
            template<typename INPUT>
                static void apply(const INPUT& in, graph_matching_instance& gmInput)
                {
                    //gmInput.no_right_nodes = std::stoul(in.string());
                }
        };

        template<> struct action< assignment > {
            template<typename INPUT>
                static void apply(const INPUT& in, graph_matching_instance& gmInput)
                {
                    std::istringstream iss(in.string());
                    std::size_t assignmentsno; iss >> assignmentsno;
                    std::size_t left_node; iss >> left_node;
                    std::size_t right_node; iss >> right_node;
                    double cost; iss >> cost;

                    gmInput.linear_assignments.push_back({left_node, right_node, cost});
                }
        };
        template<> struct action< quadratic_pot > {
            template<typename INPUT>
                static void apply(const INPUT & in, graph_matching_instance& gmInput)
                {
                    std::istringstream iss(in.string());
                    std::size_t assignments1; iss >> assignments1;
                    std::size_t assignments2; iss >> assignments2;
                    double cost; iss >> cost;

                    if(assignments1 >= gmInput.linear_assignments.size())
                        throw std::runtime_error("assignment number in quadratic infeasible: too large");
                    if(assignments2 >= gmInput.linear_assignments.size())
                        throw std::runtime_error("assignment number in quadratic infeasible: too large");

                    // check that quadratic assingment is feasible
                    const std::size_t left_1 = gmInput.linear_assignments[assignments1].i;
                    const std::size_t right_1 = gmInput.linear_assignments[assignments1].j;
                    const std::size_t left_2 = gmInput.linear_assignments[assignments2].i;
                    const std::size_t right_2 = gmInput.linear_assignments[assignments2].j;

                    // It seems that many graph matching instances have quadratic potentials that point to the same node.
                    // Ignore those instead of throwing
                    if(left_1 == left_2 && left_1 != graph_matching_instance::no_assignment)
                    {
                        return;
                        // throw std::runtime_error("assignments in quadratic infeasible: origin from same node");
                    }
                    if(right_1 == right_2 && right_1 != graph_matching_instance::no_assignment)
                    {
                        return;
                        // throw std::runtime_error("assignments in quadratic infeasible: point to same node");
                    }

                    gmInput.quadratic_assignments.push_back({{left_1, left_2}, {right_1, right_2}, cost});
                }
        };
    }
}
