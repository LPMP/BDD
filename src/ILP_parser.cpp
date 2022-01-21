#include "ILP_parser.h"
#include <tao/pegtl.hpp>
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
        //using opt_whitespace = tao::pegtl::star<tao::pegtl::sor<parsing::opt_whitespace, tao::pegtl::eol>>;
        using parsing::opt_whitespace;
        using parsing::mand_whitespace;
        using parsing::opt_invisible;
        using parsing::mand_invisible;
        using parsing::real_number;

        struct comment_line : tao::pegtl::seq<tao::pegtl::string<'\\'>, tao::pegtl::until<tao::pegtl::eol>> {};

        struct min_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'M','i','n','i','m','i','z','e'>, opt_whitespace, tao::pegtl::eol> {};

        //struct sign : tao::pegtl::sor<tao::pegtl::string<'+'>, tao::pegtl::string<'-'>> {};

        struct sign : tao::pegtl::one< '+', '-' > {};

        /*
        struct variable_name : tao::pegtl::seq< 
                               tao::pegtl::alpha, 
                               tao::pegtl::star< tao::pegtl::sor< tao::pegtl::range<'a', 'z'>, tao::pegtl::range<'A','Z'>, tao::pegtl::range<'0','9'>, tao::pegtl::one< '_', '-', '(', ')', '{', '}', ',' > > >
                               >{};
                               
                               */
        struct term_identifier : tao::pegtl::seq<
                                 tao::pegtl::plus<tao::pegtl::sor<tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','>, tao::pegtl::string<';'>, tao::pegtl::string<'@'>, tao::pegtl::string<'['>, tao::pegtl::string<']'>, tao::pegtl::string<'#'>, tao::pegtl::string<'.'> > >
                                 > {};


        struct variable_name : tao::pegtl::seq< 
                               tao::pegtl::alpha, 
                               tao::pegtl::star< tao::pegtl::sor< tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','>, tao::pegtl::string<'#'>, tao::pegtl::string<';'>, tao::pegtl::string<'['>, tao::pegtl::string<']'>, tao::pegtl::string<'.'> > > 
                                   > {};

        // TODO: remove?
        struct coefficient : tao::pegtl::opt<real_number> {};

        struct coefficient_variable : tao::pegtl::seq<coefficient, opt_whitespace, variable_name> {};

        struct objective_coefficient : real_number {};
        struct objective_variable : variable_name {};
        struct objective_term : tao::pegtl::seq< tao::pegtl::opt<sign, opt_whitespace, tao::pegtl::opt<tao::pegtl::eol>, opt_whitespace >, tao::pegtl::opt<objective_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace>, objective_variable> {};
        struct objective_constant : real_number {};
        struct subject_to : tao::pegtl::istring<'S','u','b','j','e','c','t',' ','T','o'> {};
        struct objective_line : tao::pegtl::seq< tao::pegtl::not_at<subject_to>, opt_whitespace, tao::pegtl::seq<tao::pegtl::opt<tao::pegtl::seq<term_identifier, tao::pegtl::string<':'>>>, opt_whitespace, tao::pegtl::star<opt_whitespace, objective_term>, opt_whitespace, tao::pegtl::opt<objective_constant>>, opt_whitespace, tao::pegtl::eol> {};

        struct subject_to_line : tao::pegtl::seq<opt_whitespace, subject_to, opt_whitespace, tao::pegtl::eol> {};

        struct inequality_type : tao::pegtl::sor<tao::pegtl::string<'<','='>, tao::pegtl::string<'>','='>, tao::pegtl::string<'='>> {};

        struct new_inequality_identifier : tao::pegtl::seq<term_identifier, opt_whitespace, tao::pegtl::string<':'>> {};

        struct new_inequality : tao::pegtl::seq<opt_whitespace, tao::pegtl::not_at<tao::pegtl::sor<tao::pegtl::string<'E','n','d'>, tao::pegtl::string<'B','o','u','n','d','s'>, tao::pegtl::string<'B','i','n','a','r','i','e','s'>, tao::pegtl::string<'C','o','a','l','e','s','c','e'>>>, tao::pegtl::opt<new_inequality_identifier>, opt_whitespace> {};

        struct inequality_coefficient : real_number {};
        struct inequality_variable : variable_name {};
        struct inequality_monomial : tao::pegtl::seq<inequality_variable, tao::pegtl::star<opt_whitespace, tao::pegtl::sor<tao::pegtl::string<'*'>, mand_whitespace>, opt_whitespace, inequality_variable>> {};
        struct inequality_term : tao::pegtl::seq< tao::pegtl::opt<sign, opt_whitespace>, tao::pegtl::opt<inequality_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace>, inequality_monomial> {};
        struct right_hand_side : real_number {};

        struct inequality_line : tao::pegtl::seq< new_inequality, 
            tao::pegtl::star<opt_whitespace, inequality_term, opt_whitespace, tao::pegtl::opt<tao::pegtl::eol>>,
            opt_whitespace, inequality_type, opt_whitespace, right_hand_side, opt_whitespace, tao::pegtl::eol> {};

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

        struct end_line : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'E','n','d'>, opt_whitespace, tao::pegtl::eolf> {};

        struct grammar : tao::pegtl::seq<
                         tao::pegtl::star<comment_line>,
                         min_line,
                         tao::pegtl::star<objective_line>,
                         subject_to_line,
                         tao::pegtl::star<inequality_line>,
                         //tao::pegtl::opt<coalesce_begin, tao::pegtl::star<coalesce_line>>,
                         bounds_begin, // ignore everything after bounds (variables are assumed to be binary)
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
        };

        template<> struct action< sign > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    if(in.string() == "+") {
                    } else if(in.string() == "-") {
                        tmp.objective_coeff *= -1.0;
                        tmp.constraint_coeff *= -1;
                    } else
                        throw std::runtime_error("sign not recognized");
                }
        };

        template<> struct action< objective_coefficient > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    tmp.objective_coeff *= std::stod(in.string());
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

        template<> struct action< inequality_coefficient > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    tmp.constraint_coeff *= std::stoi(in.string());
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

        template<> struct action< right_hand_side > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, tmp_storage& tmp)
                {
                    const double val = std::stod(in.string());
                    i.set_right_hand_side(val);
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

        ILP_input parse_file(const std::string& filename)
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            ILP_input ilp;
            tmp_storage tmp;
            tao::pegtl::file_input input(filename);
            if(!tao::pegtl::parse<grammar, action>(input, ilp, tmp))
                throw std::runtime_error("could not read input file " + filename);
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
            return ilp;
        }

    } 

}
