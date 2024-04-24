#include "OPB_parser.h"
#include <tao/pegtl.hpp>
#include "pegtl_parse_rules.h"
#include "ILP_input.h"
#include "time_measure_util.h" 

namespace LPMP { 

    std::string remove_whitespace(const std::string& str)
    {
        std::string result = str;
        result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());
        return result;
    }

    namespace OPB_parser {

        // import basic parsers
        using parsing::opt_whitespace;
        using parsing::mand_whitespace;
        using parsing::real_number;
        using parsing::positive_integer;

        struct comment_line : tao::pegtl::seq<tao::pegtl::string<'*'>, tao::pegtl::until<tao::pegtl::eol>> {};

        struct min_begin : tao::pegtl::seq<opt_whitespace, tao::pegtl::string<'m','i','n',':'>, opt_whitespace> {};

        //struct sign : tao::pegtl::sor<tao::pegtl::string<'+'>, tao::pegtl::string<'-'>> {};

        struct sign : tao::pegtl::one< '+', '-' > {};

        struct variable_name : tao::pegtl::seq< 
                               tao::pegtl::alpha, 
                               tao::pegtl::star< tao::pegtl::sor< tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','> > > 
                               > {};

        struct variable_coefficient : tao::pegtl::opt<real_number> {};

        struct variable : tao::pegtl::seq<variable_coefficient, opt_whitespace, variable_name> {};

        struct first_objective_coefficient : tao::pegtl::seq<tao::pegtl::opt<sign, opt_whitespace>, tao::pegtl::opt<real_number>> {};
        struct subsequent_objective_coefficient : tao::pegtl::seq<sign, tao::pegtl::opt<opt_whitespace, real_number>> {};
        struct objective_variable : variable_name {};
        struct first_objective_term : tao::pegtl::seq<first_objective_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, objective_variable> {};
        struct subsequent_objective_term : tao::pegtl::seq< subsequent_objective_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, objective_variable> {};
        struct end_objective : tao::pegtl::string<';'> {};
        struct objective_line : tao::pegtl::seq< min_begin, tao::pegtl::not_at<tao::pegtl::string<';'>>, opt_whitespace, first_objective_term, tao::pegtl::star<opt_whitespace, subsequent_objective_term>, opt_whitespace, end_objective, opt_whitespace, tao::pegtl::eol> {};

        struct inequality_type : tao::pegtl::sor<tao::pegtl::string<'<','='>, tao::pegtl::string<'>','='>, tao::pegtl::string<'='>> {};

        struct inequality_identifier : tao::pegtl::seq<
        tao::pegtl::plus<tao::pegtl::sor<tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','> > >
        > {};
        //struct inequality_coefficient : tao::pegtl::seq<tao::pegtl::digit, tao::pegtl::star<tao::pegtl::digit>> {};
        struct first_inequality_coefficient : tao::pegtl::seq<tao::pegtl::opt<sign, opt_whitespace>, tao::pegtl::opt<positive_integer>> {};
        struct subsequent_inequality_coefficient : tao::pegtl::seq<sign, tao::pegtl::opt<opt_whitespace, positive_integer>> {};
        struct inequality_variable : variable_name {};
        struct end_inequality : tao::pegtl::string<';'> {};
        struct first_inequality_term : tao::pegtl::seq< first_inequality_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, inequality_variable> {};
        struct subsequent_inequality_term : tao::pegtl::seq< subsequent_inequality_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace, inequality_variable> {};

        struct right_hand_side : tao::pegtl::seq< tao::pegtl::opt<sign>, tao::pegtl::digit, tao::pegtl::star<tao::pegtl::digit> > {};

        struct inequality_line : tao::pegtl::seq< opt_whitespace, first_inequality_term, tao::pegtl::star<opt_whitespace, subsequent_inequality_term, opt_whitespace, tao::pegtl::opt<tao::pegtl::eol>>,
            opt_whitespace, inequality_type, opt_whitespace, right_hand_side, opt_whitespace, end_inequality, opt_whitespace, tao::pegtl::eol> {};

        struct grammar : tao::pegtl::seq<
                         tao::pegtl::star<comment_line>,
                         objective_line,
                         tao::pegtl::star<inequality_line>,
                         tao::pegtl::until<tao::pegtl::eof>> {};

        template< typename Rule >
            struct action
            : tao::pegtl::nothing< Rule > {};

        using constraint = ILP_input::constraint;
        using objective = std::vector<std::tuple<double, std::string>>;

        using tmp_storage = typename ILP_input::constraint;
        //struct tmp_storage {
        //    typename ILP_input::constraint constraint
        //    double objective_coeff = 1.0;
        //    int constraint_coeff = 1;
        //    std::string inequality_identifier = "";
        //};

        /*
        template<> struct action< objective_sign > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    if(in.string() == "+") {
                    } else if(in.string() == "-") {
                    //if(in.string() == "+") {
                    //} else if(in.string() == "-") {
                    //    tmp.coefficients.push_back(-1);
                    //    tmp.objective_coeff *= -1.0;
                    //    tmp.constraint_coeff *= -1;
                    } else
                        throw std::runtime_error("sign not recognized");
                }
        };
        */

       double parse_real_coefficient(std::string str)
       {
           str = remove_whitespace(str);
           if (str == "")
               return 1.0;
           else if (str == "-")
               return -1.0;
           else if (str == "+")
               return 1.0;
           else
               return std::stod(str);
       }

       double parse_integer_coefficient(std::string str)
       {
           str = remove_whitespace(str);
           if (str == "")
               return 1.0;
           else if (str == "-")
               return -1.0;
           else if (str == "+")
               return 1.0;
           else
               return std::stoi(str);
       }

       template <> struct action<first_objective_coefficient> {
           template <typename INPUT>
           static void apply(const INPUT &in, ILP_input &i, objective &obj, constraint &constr)
           {
               obj.push_back({parse_real_coefficient(in.string()), ""});
           }
       };

       template <> struct action<subsequent_objective_coefficient> {
           template <typename INPUT>
           static void apply(const INPUT &in, ILP_input &i, objective &obj, constraint &constr)
           {
               obj.push_back({parse_real_coefficient(in.string()), ""});
           }
       };

        template<> struct action< objective_variable > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    const std::string var = in.string();
                    assert(obj.size() > 0);
                    std::get<1>(obj.back()) = var;
                }
        };

        template<> struct action< end_objective > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    for(const auto [coeff, var] : obj)
                        i.add_to_objective(coeff, var);
                    obj.clear();
                }
        };

        template<> struct action< first_inequality_coefficient > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    constr.coefficients.push_back(parse_integer_coefficient(in.string()));
                }
        };

        template<> struct action< subsequent_inequality_coefficient > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    constr.coefficients.push_back(parse_integer_coefficient(in.string()));
                }
        };

        template<> struct action< inequality_variable > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    const std::string var = in.string();
                    const size_t var_idx = i.get_or_create_variable_index(var);
                    constr.monomials.push_back(&var_idx, &var_idx+1);
                    assert(constr.coefficients.size() == constr.monomials.size());
                }
        };

        template<> struct action< inequality_type > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    const std::string ineq = in.string();
                    if(ineq == "=")
                        constr.ineq = ILP_input::inequality_type::equal;
                    else if(ineq == "<=")
                        constr.ineq = ILP_input::inequality_type::smaller_equal;
                    else if(ineq == ">=")
                        constr.ineq = ILP_input::inequality_type::greater_equal;
                    else
                        throw std::runtime_error("inequality type not recognized");
                }
        };

        template<> struct action< right_hand_side > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    const double val = std::stod(in.string());
                    constr.right_hand_side = val;
                }
        };

        template<> struct action< end_inequality > {
            template<typename INPUT>
                static void apply(const INPUT & in, ILP_input& i, objective& obj, constraint& constr)
                {
                    i.add_constraint(constr);
                    constr = constraint();
                }
        };

        ILP_input parse_file(const std::string& filename)
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            ILP_input ilp;
            constraint constr;
            objective obj;
            tao::pegtl::file_input input(filename);
            if(!tao::pegtl::parse<grammar, action>(input, ilp, obj, constr))
                throw std::runtime_error("could not read input file " + filename);
            return ilp;
        }

        ILP_input parse_string(const std::string& input_string)
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            ILP_input ilp;
            constraint constr;
            objective obj;
            tao::pegtl::string_input input(input_string, "OPB input");

            if(!tao::pegtl::parse<grammar, action>(input, ilp, obj, constr))
                throw std::runtime_error("could not read input:\n" + input_string);
            return ilp;
        }

    } 

}

