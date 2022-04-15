#include <iostream>
#include "specialized_solvers/mrf_input.h"
#include <tao/pegtl.hpp>
#include <variant>
#include "pegtl_parse_rules.h"
#include "time_measure_util.h"

namespace LPMP {

    namespace mrf_uai_input {

        using parsing::mand_whitespace; 
        using parsing::opt_whitespace;  
        using parsing::opt_invisible;  
        using parsing::positive_integer;
        using parsing::real_number;

        struct mrf_input_helper {
            std::vector<size_t> nr_labels;
            using clique_scope_type = std::variant< size_t, std::array<size_t,2>, std::vector<size_t> >;
            std::vector<clique_scope_type> clique_scopes;
            size_t number_of_cliques;

            std::vector<double> current_function_table;
            size_t current_function_table_size;
            size_t current_clique_number = 0;
            size_t current_pairwise_clique_number = 0;
        };

        struct number_of_variables : tao::pegtl::seq< opt_whitespace, positive_integer, opt_whitespace > {};
        // vector of integers denoting how many labels each variable has
        struct nr_labels : tao::pegtl::seq< opt_whitespace, positive_integer, opt_whitespace > {};
        struct allocate_data : tao::pegtl::seq<> {};
        struct number_of_cliques : tao::pegtl::seq< opt_whitespace, positive_integer, opt_whitespace> {};

        // first is the number of variables in the clique, then the actual variables.
        // the clique_scopes should match number_of_clique_lines, each line consisting of a sequence of integers
        struct new_clique_scope : tao::pegtl::seq< positive_integer > {};
        struct clique_scope : tao::pegtl::seq< positive_integer > {};
        struct clique_scope_line : tao::pegtl::seq< opt_whitespace, new_clique_scope, tao::pegtl::plus< opt_whitespace, clique_scope >, opt_whitespace, tao::pegtl::eol > {};

        struct clique_scopes_end : tao::pegtl::maybe_nothing
        {
            template< 
                tao::pegtl::apply_mode A,
                tao::pegtl::rewind_mode M,
                template< typename ... > class Action,
                template< typename ... > class Control,
                typename Input
                    >
                    static bool match(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                    {
                        return input_helper.number_of_cliques == input_helper.clique_scopes.size();
                    }

        };

        struct clique_scopes : tao::pegtl::until< clique_scopes_end, clique_scope_line > {};

        // a function table is begun by number of entries and then a list of real numbers. Here we record all the values in the real stack
        // do zrobienia: treat whitespace

        struct new_function_table : tao::pegtl::seq< positive_integer > {};
        struct function_table_entry : tao::pegtl::seq< real_number > {};
        struct function_tables_end
        {
            template< 
                tao::pegtl::apply_mode A,
                tao::pegtl::rewind_mode M,
                template< typename ... > class Action,
                template< typename ... > class Control,
                typename Input
                    >
                    static bool match(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                    {
                    return input_helper.number_of_cliques == input_helper.current_clique_number;
                    }

        };
        struct function_table_end
        {
            template<typename TABLE>
                static void write_matrix_into_instance(mrf_input& input, const TABLE& table, const mrf_input_helper& input_helper, const std::array<std::size_t,2> vars)
                {
                    assert(input.nr_labels(vars[0])*input.nr_labels(vars[1]) == table.size());
                    for(std::size_t i=0; i<input.nr_labels(vars[0]); ++i) {
                        for(std::size_t j=0; j<input.nr_labels(vars[1]); ++j) {
                            input.pairwise(input_helper.current_pairwise_clique_number, {i, j}) = table[i*input.nr_labels(vars[1]) + j];
                        }
                    } 
                }
            template<typename VECTOR>
                static void write_vector_into_instance(mrf_input& input, const VECTOR& table, const size_t var)
                {
                    assert(input.nr_labels(var) == table.size());
                    for(std::size_t i=0; i<table.size(); ++i) {
                        assert(input.unary(var,i) == 0.0);
                        input.unary(var,i) = table[i];
                    } 
                }

            template< 
                tao::pegtl::apply_mode A,
                tao::pegtl::rewind_mode M,
                template< typename ... > class Action,
                template< typename ... > class Control,
                typename Input
                    >
                    static bool match(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                    {
                    if(input_helper.current_function_table.size() == input_helper.current_function_table_size) {

                        auto& table = input_helper.current_function_table;
                        const std::size_t clique_number = input_helper.current_clique_number;
                        auto& clique_scope = input_helper.clique_scopes[clique_number];
                        // write function table into unary, pairwise or higher order table
                        if( std::holds_alternative<std::size_t>( clique_scope ) ) {
                            const auto var = std::get<std::size_t>(clique_scope);
                            write_vector_into_instance(input, table, var);
                        } else if( std::holds_alternative<std::array<std::size_t,2>>( clique_scope ) ) {
                            auto& vars = std::get<std::array<std::size_t,2>>(clique_scope);
                            write_matrix_into_instance(input, table, input_helper, vars);
                            input_helper.current_pairwise_clique_number++; 
                        } else {
                            assert(std::holds_alternative<std::vector<std::size_t>>(clique_scope));
                            assert(false); // not implemented yet
                        } 

                        table.clear();
                        input_helper.current_clique_number++;
                        return true;

                    } else {
                        return false;
                    }
                    }

        };

        struct function_table : tao::pegtl::seq< new_function_table, opt_invisible, tao::pegtl::until< function_table_end, opt_invisible, function_table_entry >, opt_invisible > {};
        struct function_tables : tao::pegtl::seq< opt_invisible, tao::pegtl::until< function_tables_end, function_table >, opt_invisible > {};

        template< typename Rule >
            struct action
            : tao::pegtl::nothing< Rule > {};

        template<> struct action< number_of_variables > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    input_helper.nr_labels.reserve(std::stoul(in.string()));
                }
        };

        template<> struct action< number_of_cliques > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    input_helper.number_of_cliques = std::stoul(in.string()); 
                }
        };

        template<> struct action< nr_labels > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    input_helper.nr_labels.push_back(std::stoul(in.string()));
                }
        };

        template<> struct action< allocate_data > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    // allocate space for unary and pairwise factors
                    input.unaries_.resize(input_helper.nr_labels.begin(), input_helper.nr_labels.end());
                    // we must write 0 into unaries, since not all unary potentials must be present
                    for(std::size_t i=0; i<input.nr_variables(); ++i) {
                        for(std::size_t j=0; j<input.nr_labels(i); ++j) {
                            input.unary(i,j) = 0.0;
                        }
                    }

                    std::vector<size_t> pairwise_size;
                    pairwise_size.reserve(input.pairwise_variables_.size());
                    for(const auto pairwise_variables : input.pairwise_variables_) {
                        pairwise_size.push_back( input.nr_labels(pairwise_variables[0]) * input.nr_labels(pairwise_variables[1]) );
                    }

                    input.pairwise_.resize(pairwise_size.begin(), pairwise_size.end());
                } 
        };

        template<> struct action< new_clique_scope > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    const size_t arity = std::stoul(in.string());
                    assert(arity == 1 || arity == 2); // higher order not yet supported
                    mrf_input_helper::clique_scope_type clique_scope;
                    if(arity == 1) {
                        clique_scope = std::size_t(std::numeric_limits<std::size_t>::max());
                    } else if(arity == 2) {
                        clique_scope = std::array<std::size_t,2>( {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()} );
                    } else {
                        assert(false);
                    }
                    input_helper.clique_scopes.push_back( clique_scope );
                }
        };

        template<> struct action< clique_scope > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    const auto var_no = std::stoul(in.string());
                    assert(var_no < input_helper.nr_labels.size());
                    auto& clique_scope = input_helper.clique_scopes.back();
                    if( std::holds_alternative<std::size_t>( clique_scope ) ) {
                        std::get<std::size_t>(clique_scope) = var_no;
                    } else if( std::holds_alternative<std::array<std::size_t,2>>( clique_scope ) ) {
                        auto& pairwise_idx = std::get<std::array<std::size_t,2>>(clique_scope);
                        if(pairwise_idx[0] == std::numeric_limits<std::size_t>::max()) {
                            pairwise_idx[0] = var_no;
                        } else {
                            assert(pairwise_idx[1] == std::numeric_limits<std::size_t>::max());
                            pairwise_idx[1] = var_no;
                            input.pairwise_variables_.push_back({pairwise_idx[0], pairwise_idx[1]});
                        }
                    } else {
                        assert(std::holds_alternative<std::vector<std::size_t>>(clique_scope));
                        std::get<std::vector<std::size_t>>(clique_scope).push_back(var_no);
                    }
                }
        };
        template<> struct action< new_function_table > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    input_helper.current_function_table.clear();
                    input_helper.current_function_table_size  = std::stoul(in.string());
                    input_helper.current_function_table.reserve( input_helper.current_function_table_size );
                }
        };
        template<> struct action< function_table_entry > {
            template<typename Input>
                static void apply(const Input& in, mrf_input& input, mrf_input_helper& input_helper) 
                {
                    input_helper.current_function_table.push_back(std::stod(in.string()));
                }
        };

        struct grammar :
            tao::pegtl::seq<
            opt_whitespace, tao::pegtl::string<'M','A','R','K','O','V'>, opt_whitespace, tao::pegtl::eol,
            number_of_variables, tao::pegtl::eol,
            tao::pegtl::plus< nr_labels >, tao::pegtl::eol,
            number_of_cliques, tao::pegtl::eol,
            clique_scopes,
            opt_invisible,
            allocate_data,
            function_tables
            > {};
    }

    mrf_input parse_mrf_uai_file(const std::string& filename)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        mrf_input mrf;
        mrf_uai_input::mrf_input_helper input_helper;
        tao::pegtl::file_input input(filename);

        if(!tao::pegtl::parse<mrf_uai_input::grammar, mrf_uai_input::action>(input, mrf, input_helper))
            throw std::runtime_error("[mrf uai parser] Could not read file " + filename);

        std::cout << "[mrf uai parser] problem has " << mrf.nr_variables() << " variables and " << mrf.nr_pairwise_potentials() << " pairwise potentials\n";
        return mrf; 
    }

    mrf_input parse_mrf_uai_string(const std::string& uai_string)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        mrf_input mrf;
        mrf_uai_input::mrf_input_helper input_helper;
        tao::pegtl::string_input input(uai_string, "mrf uai input");

        if(!tao::pegtl::parse<mrf_uai_input::grammar, mrf_uai_input::action>(input, mrf, input_helper))
            throw std::runtime_error("[mrf uai parser] Could not read string");

        std::cout << "[mrf uai parser] problem has " << mrf.nr_variables() << " variables and " << mrf.nr_pairwise_potentials() << " pairwise potentials\n";
        return mrf; 
    }


}
