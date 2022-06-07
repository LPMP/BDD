#include "ILP_parser.h"
#include <tao/pegtl.hpp>
#include <unordered_set>
#include <algorithm>
#include "pegtl_parse_rules.h"
#include "ILP_input.h"
#include "time_measure_util.h" 

namespace LPMP { 

    inline std::string& trim(std::string & str)
    {
        // right trim
        while (str.length () > 0 && (str [str.length ()-1] == ' ' || str [str.length ()-1] == '\t'))
            str.erase (str.length ()-1, 1);

        // left trim
        while (str.length () > 0 && (str [0] == ' ' || str [0] == '\t'))
            str.erase (0, 1);
        return str;
    }

    namespace ILP_parser {

        // import basic parsers
        using parsing::opt_whitespace;
        using parsing::mand_whitespace;
        using parsing::opt_invisible;
        using parsing::mand_invisible;
        using parsing::real_number;

        struct comment_line : tao::pegtl::seq<tao::pegtl::string<'\\'>, tao::pegtl::until<tao::pegtl::eol>> {};

        struct min_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'M','i','n','i','m','i','z','e'>, opt_whitespace, tao::pegtl::eol> {};

        struct sign : tao::pegtl::one< '+', '-' > {};

        struct term_identifier : tao::pegtl::seq<
                                 tao::pegtl::plus<tao::pegtl::sor<tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','>, tao::pegtl::string<';'>, tao::pegtl::string<'@'>, tao::pegtl::string<'['>, tao::pegtl::string<']'>, tao::pegtl::string<'#'>, tao::pegtl::string<'.'>, tao::pegtl::string<'\''> > >
                                 > {};


        struct variable_name : tao::pegtl::seq< 
                               tao::pegtl::alpha, 
                               tao::pegtl::star< tao::pegtl::sor< tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','>, tao::pegtl::string<'#'>, tao::pegtl::string<';'>, tao::pegtl::string<'['>, tao::pegtl::string<']'>, tao::pegtl::string<'.'>, tao::pegtl::string<'\''> > > 
                                   > {};

        // TODO: allow for case that no objective or just a constant is given.
        struct objective_variable : variable_name {};

        struct first_objective_coefficient : tao::pegtl::seq<tao::pegtl::opt<sign>, opt_whitespace, tao::pegtl::opt<real_number> > {};
        struct first_objective_term : tao::pegtl::seq< tao::pegtl::opt<first_objective_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>>, opt_whitespace, objective_variable> {};

        struct subsequent_objective_coefficient_1 : tao::pegtl::seq<sign, opt_whitespace, real_number > {};
        struct subsequent_objective_term_1 : tao::pegtl::seq< subsequent_objective_coefficient_1, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, objective_variable> {};

        struct subsequent_objective_coefficient_2 : sign {};
        struct subsequent_objective_term_2 : tao::pegtl::seq< subsequent_objective_coefficient_2, opt_whitespace, objective_variable > {};

        struct subsequent_objective_term : tao::pegtl::sor< subsequent_objective_term_1, subsequent_objective_term_2 > {};

        struct objective_constant : tao::pegtl::seq<sign, opt_whitespace, real_number> {};

        struct subject_to : tao::pegtl::istring<'S','u','b','j','e','c','t',' ','T','o'> {};

        // TODO: use not_at instead of until
        struct objective : tao::pegtl::until< subject_to, 
                                              tao::pegtl::seq< opt_invisible, first_objective_term, 
                                                               tao::pegtl::star< opt_invisible, subsequent_objective_term >
                                                             >,
                                              opt_invisible, tao::pegtl::opt< objective_constant, opt_invisible > 
                                            >
        {};

        struct subject_to_line : tao::pegtl::seq<opt_whitespace, subject_to, opt_whitespace, tao::pegtl::eol> {};

        struct inequality_type : tao::pegtl::sor<tao::pegtl::string<'<','='>, tao::pegtl::string<'>','='>, tao::pegtl::string<'='>> {};

        struct new_inequality_identifier : tao::pegtl::seq<term_identifier, opt_whitespace, tao::pegtl::string<':'>> {};

        struct new_inequality : tao::pegtl::seq<opt_whitespace, tao::pegtl::not_at<tao::pegtl::sor<tao::pegtl::string<'E','n','d'>, tao::pegtl::string<'B','o','u','n','d','s'>, tao::pegtl::string<'B','i','n','a','r','i','e','s'>, tao::pegtl::string<'C','o','a','l','e','s','c','e'>>>, tao::pegtl::opt<new_inequality_identifier>, opt_whitespace> {};

        // TODO: fix whitespace issue with monomial
        struct inequality_variable : variable_name {};
        struct inequality_monomial : tao::pegtl::seq<inequality_variable, tao::pegtl::star<opt_whitespace, tao::pegtl::sor<tao::pegtl::string<'*'> >, opt_whitespace, inequality_variable>> {};

        // TODO: replace real_number by integer for inequalities
        // TODO: first term has * only if coefficient is present, same for objective
        struct first_inequality_coefficient : tao::pegtl::seq<tao::pegtl::opt<sign>, opt_whitespace, tao::pegtl::opt<real_number>> {};
        struct first_inequality_term : tao::pegtl::seq<first_inequality_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, inequality_monomial> {};

        struct subsequent_inequality_coefficient_1 : sign {};
        struct subsequent_inequality_coefficient_2 : tao::pegtl::seq<sign, opt_invisible, real_number> {};
        struct subsequent_inequality_term_1 : tao::pegtl::seq< subsequent_inequality_coefficient_1, opt_invisible, inequality_monomial> {};
        struct subsequent_inequality_term_2 : tao::pegtl::seq< subsequent_inequality_coefficient_2, opt_invisible, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, inequality_monomial> {};
        struct subsequent_inequality_term : tao::pegtl::sor<subsequent_inequality_term_1, subsequent_inequality_term_2> {};

        struct lhs_inequality : tao::pegtl::seq<first_inequality_term, tao::pegtl::star<opt_invisible, subsequent_inequality_term>> {};

        struct rhs_inequality : tao::pegtl::seq<tao::pegtl::opt<sign>, opt_invisible, real_number> {};

        struct inequality_line : tao::pegtl::seq<
                                 new_inequality, 
                                 opt_invisible,
                                 lhs_inequality,
                                 opt_invisible,
                                 inequality_type,
                                 opt_invisible,
                                 rhs_inequality,
                                 opt_whitespace,
                                 tao::pegtl::eol
                                 > {};

        struct coalesce_begin : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'C','o','a','l','e','s','c','e'>, opt_whitespace, tao::pegtl::eol> {};

        struct coalesce_identifier : term_identifier {};
        struct coalesce_line : tao::pegtl::seq<opt_whitespace, coalesce_identifier, tao::pegtl::plus<opt_whitespace, coalesce_identifier>, opt_whitespace, tao::pegtl::eol> {};

        struct bounds_begin : tao::pegtl::opt<tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'B','o','u','n','d','s'>, opt_whitespace, tao::pegtl::eol>> {};
        struct generals_begin : tao::pegtl::opt<tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'G','e','n','e','r','a','l','s'>, opt_whitespace, tao::pegtl::eol>> {};

        struct binaries_begin : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'B','i','n','a','r','i','e','s'>, opt_whitespace, tao::pegtl::eol> {};
        struct binary_variable : variable_name {};
        struct list_of_binary_variables : tao::pegtl::seq< 
            tao::pegtl::star<opt_whitespace, tao::pegtl::opt<tao::pegtl::eol>, opt_whitespace, tao::pegtl::not_at< tao::pegtl::string<'E','n','d'> >, binary_variable>,
            opt_whitespace, tao::pegtl::eol > {};

        struct binaries : tao::pegtl::opt<binaries_begin, list_of_binary_variables> {};

        struct bounds_var_fixation_line : tao::pegtl::seq<opt_whitespace, variable_name, opt_whitespace, tao::pegtl::string<'='>, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'0'>, tao::pegtl::string<'1'>>, opt_whitespace, tao::pegtl::eol> {};
        struct bounds_var_smaller_equal_line : tao::pegtl::seq<opt_whitespace, variable_name, opt_whitespace, tao::pegtl::string<'<','='>, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'0'>, tao::pegtl::string<'1'>>, opt_whitespace, tao::pegtl::eol> {};
        struct bounds_var_greater_equal_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'0'>, tao::pegtl::string<'1'>>, opt_whitespace, tao::pegtl::string<'<','='>, opt_whitespace,  variable_name, opt_whitespace, tao::pegtl::eol> {};
        struct bounds_var_lb_ub_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'0'>, tao::pegtl::string<'1'>>, opt_whitespace, tao::pegtl::string<'<','='>, opt_whitespace,  variable_name, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'0'>, tao::pegtl::string<'1'>>, tao::pegtl::string<'<','='>, opt_whitespace, opt_whitespace, tao::pegtl::eol> {};

        struct end_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'E','n','d'>, opt_invisible, tao::pegtl::eolf> {};

        struct grammar : tao::pegtl::seq<
                         tao::pegtl::star<comment_line>,
                         min_line,
                         objective,
                         opt_invisible,
                         //subject_to_line,
                         opt_invisible,
                         tao::pegtl::star<inequality_line, opt_whitespace>,
                         //tao::pegtl::opt<coalesce_begin, tao::pegtl::star<coalesce_line>>,
                         bounds_begin, // ignore everything after bounds (variables are assumed to be binary)
                         tao::pegtl::star<tao::pegtl::sor<
                            bounds_var_fixation_line,
                            bounds_var_smaller_equal_line,
                            bounds_var_greater_equal_line,
                            bounds_var_lb_ub_line
                         >>,
                         //binaries,
                         //end_line> {};
                         tao::pegtl::until<end_line>> {};

        template< typename Rule >
            struct action
            : tao::pegtl::nothing< Rule > {};

        struct tmp_storage {
            double objective_coeff = 1.0;
            int constraint_coeff = 1;
            std::vector<size_t> constraint_monomial;
            std::string inequality_identifier = "";
            std::vector<std::string> coalesce_identifiers;
            std::unordered_set<size_t> zero_fixations;
            std::unordered_set<size_t> one_fixations;
        };

        template<> struct action< first_objective_coefficient > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::string stripped = in.string();
                    stripped.erase(std::remove_if(stripped.begin(), stripped.end(), ::isspace), stripped.end());
                    if(stripped == "-")
                        tmp.objective_coeff = -1.0;
                    else if(stripped == "+" || stripped == "")
                        tmp.objective_coeff = 1.0;
                    else
                        tmp.objective_coeff = std::stod(stripped);
                }
        };

        template<> struct action< subsequent_objective_coefficient_1 > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::string stripped = in.string();
                    stripped.erase(std::remove_if(stripped.begin(), stripped.end(), ::isspace), stripped.end());
                    tmp.objective_coeff = std::stod(stripped);
                }
        };

        template<> struct action< subsequent_objective_coefficient_2 > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    if(in.string() == "-")
                        tmp.objective_coeff = -1.0;
                    else if(in.string() == "+")
                        tmp.objective_coeff = 1.0;
                    else
                        throw std::runtime_error("only {+|-} allowed");
                }
        };

        template<> struct action< objective_variable > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    const std::string var = in.string();
                    i.add_to_objective(tmp.objective_coeff, var);
                    tmp = tmp_storage{};
                }
        };

        template<> struct action< objective_constant > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::string stripped = in.string();
                    stripped.erase(std::remove_if(stripped.begin(), stripped.end(), ::isspace), stripped.end());
                    const double constant = std::stod(stripped);
                    std::cout << "constant = " << constant << "\n";
                    i.add_to_constant(constant);
                }
        };

        template<> struct action< new_inequality > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    i.begin_new_inequality();
                    i.set_inequality_identifier(tmp.inequality_identifier);
                    tmp = tmp_storage{};
                }
        };

        template<> struct action< new_inequality_identifier > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    tmp.inequality_identifier = in.string();
                    tmp.inequality_identifier = tmp.inequality_identifier.substr(0, tmp.inequality_identifier.size()-1);
                    trim(tmp.inequality_identifier); 
                }
        };

        template<> struct action< first_inequality_coefficient > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::string stripped = in.string();
                    stripped.erase(std::remove_if(stripped.begin(), stripped.end(), ::isspace), stripped.end());
                    if(stripped == "" || stripped == "+")
                        tmp.constraint_coeff = 1;
                    else if(stripped == "-")
                        tmp.constraint_coeff = -1;
                    else
                        tmp.constraint_coeff = std::stoi(stripped);
                }
        };

        template<> struct action< subsequent_inequality_coefficient_1 > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    if(in.string() == "-")
                        tmp.constraint_coeff = -1;
                    else if(in.string() == "+" || in.string() == "")
                        tmp.constraint_coeff = 1;
                    else
                        throw std::runtime_error("only recognize {-|+}");
                }
        };

        template<> struct action< subsequent_inequality_coefficient_2 > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::string stripped = in.string();
                    stripped.erase(std::remove_if(stripped.begin(), stripped.end(), ::isspace), stripped.end());
                    tmp.constraint_coeff = std::stoi(stripped);
                }
        };

        template<> struct action< inequality_variable > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    const std::string var = in.string();
                    if(!i.var_exists(var))
                        i.add_new_variable(var);
                    const size_t var_idx = i.get_var_index(var);
                    tmp.constraint_monomial.push_back(var_idx);
                }
        };

        template<> struct action< inequality_monomial > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    i.add_to_constraint(tmp.constraint_coeff, tmp.constraint_monomial.begin(), tmp.constraint_monomial.end());
                    tmp = tmp_storage{};
                }
        };

        template<> struct action< inequality_type > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    const std::string ineq = in.string();
                    if(ineq == "=")
                        i.set_inequality_type(ILP_input::inequality_type::equal);
                    else if(ineq == "<=")
                        i.set_inequality_type(ILP_input::inequality_type::smaller_equal);
                    else if(ineq == ">=")
                        i.set_inequality_type(ILP_input::inequality_type::greater_equal);
                    else
                        throw std::runtime_error("inequality type not recognized");
                }
        };

        template<> struct action< rhs_inequality > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::string stripped = in.string();
                    stripped.erase(std::remove_if(stripped.begin(), stripped.end(), ::isspace), stripped.end());
                    i.set_right_hand_side(std::stoi(stripped));
                }
        };

        template<> struct action< coalesce_identifier > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    tmp.coalesce_identifiers.push_back(in.string());
                }
        };

        template<> struct action< coalesce_line > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    assert(tmp.coalesce_identifiers.size() > 1);
                    i.add_constraint_group(tmp.coalesce_identifiers.begin(), tmp.coalesce_identifiers.end());
                    tmp.coalesce_identifiers.clear();
                }
        };

        template<> struct action< bounds_var_fixation_line > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::istringstream iss(in.string());

                    std::string var; iss >> var;
                    assert(i.var_exists(var));

                    std::string equality; iss >> equality;
                    assert(equality == "=");

                    std::string val_str; iss >> val_str;
                    const int val = std::stoi(val_str);
                    assert(val == 0 || val == 1);

                    if(val == 0)
                        tmp.zero_fixations.insert(i.get_var_index(var));
                    else if(val == 1)
                        tmp.one_fixations.insert(i.get_var_index(var));
                }
        };

        template<> struct action< bounds_var_smaller_equal_line > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::istringstream iss(in.string());

                    std::string var; iss >> var;
                    assert(i.var_exists(var));

                    std::string equality; iss >> equality;
                    assert(equality == "<=");

                    std::string val_str; iss >> val_str;
                    const int val = std::stoi(val_str);
                    assert(val == 0 || val == 1);

                    if(val == 0)
                        tmp.zero_fixations.insert(i.get_var_index(var));
                }
        };

        template<> struct action< bounds_var_greater_equal_line > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::istringstream iss(in.string());

                    std::string val_str; iss >> val_str;
                    const int val = std::stoi(val_str);
                    assert(val == 0 || val == 1);

                    std::string equality; iss >> equality;
                    assert(equality == "<=");

                    std::string var; iss >> var;
                    assert(i.var_exists(var));

                    if(val == 1)
                        tmp.one_fixations.insert(i.get_var_index(var));
                }
        };

        template<> struct action< bounds_var_lb_ub_line > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    std::istringstream iss(in.string());

                    std::string lb_str; iss >> lb_str;
                    const int lb = std::stoi(lb_str);
                    assert(lb == 0 || lb == 1);

                    std::string first_inequality; iss >> first_inequality;
                    assert(first_inequality == "<=");

                    std::string var; iss >> var;
                    assert(i.var_exists(var));

                    std::string second_inequality; iss >> second_inequality;
                    assert(second_inequality == "<=");

                    std::string ub_str; iss >> ub_str;
                    const int ub = std::stoi(ub_str);
                    assert(ub == 0 || ub == 1);

                    assert(lb <= ub);

                    if(lb == 1)
                        tmp.one_fixations.insert(i.get_var_index(var));
                    if(ub == 0)
                        tmp.zero_fixations.insert(i.get_var_index(var));
                }
        };

        ILP_input parse_file(const std::string& filename)
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            ILP_input ilp;
            tmp_storage tmp;
            tao::pegtl::file_input input(filename);
            if(!tao::pegtl::parse<grammar, action>(input, ilp, tmp))
                throw std::runtime_error("could not read input file " + filename);
            if(tmp.one_fixations.size() > 0 || tmp.zero_fixations.size() > 0)
                ilp = ilp.reduce(tmp.zero_fixations, tmp.one_fixations);

#ifndef NDEBUG
            if(!ilp.every_variable_in_some_ineq());
            std::cout << "[ILP parser] ILP has variables that are not present in any constraint\n";

            const size_t nr_subproblems = ilp.nr_disconnected_subproblems();
            if(nr_subproblems != 1)
                std::cout << "[ILP parser] ILP can bedivided into " << nr_subproblems << " subproblems\n";
#endif
            return ilp;
        }

        ILP_input parse_string(const std::string& input_string)
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            ILP_input ilp;
            tmp_storage tmp;
            tao::pegtl::string_input input(input_string, "ILP input");

            if(!tao::pegtl::parse<grammar, action>(input, ilp, tmp))
                throw std::runtime_error("could not read input:\n" + input_string);
            if(tmp.one_fixations.size() > 0 || tmp.zero_fixations.size() > 0)
                ilp = ilp.reduce(tmp.zero_fixations, tmp.one_fixations);

#ifndef NDEBUG
            if(!ilp.every_variable_in_some_ineq());
            std::cout << "[ILP parser] ILP has variables that are not present in any constraint\n";

            const size_t nr_subproblems = ilp.nr_disconnected_subproblems();
            if(nr_subproblems != 1)
                std::cout << "[ILP parser] ILP can bedivided into " << nr_subproblems << " subproblems\n";
#endif
            return ilp;
        }

    } 

}
