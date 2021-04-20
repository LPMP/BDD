#include "OPB_parser.h"
#include <tao/pegtl.hpp>
#include "pegtl_parse_rules.h"
#include "ILP_input.h"
#include "time_measure_util.h" 

namespace LPMP { 

    std::string& trim(std::string & str)
    {
        // right trim
        while (str.length () > 0 && (str [str.length ()-1] == ' ' || str [str.length ()-1] == '\t'))
            str.erase (str.length ()-1, 1);

        // left trim
        while (str.length () > 0 && (str [0] == ' ' || str [0] == '\t'))
            str.erase (0, 1);
        return str;
    }

    namespace OPB_parser {

        // import basic parsers
        using parsing::opt_whitespace;
        using parsing::mand_whitespace;
        using parsing::real_number;

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

        struct weighted_sum_of_variables : tao::pegtl::seq< opt_whitespace, variable, tao::pegtl::star< opt_whitespace, sign, opt_whitespace, variable >>{};

        struct objective_coefficient : real_number {};
        struct objective_variable : variable_name {};
        struct objective_term : tao::pegtl::seq< tao::pegtl::opt<sign, opt_whitespace>, tao::pegtl::opt<objective_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace>, objective_variable> {};
        struct objective_line : tao::pegtl::seq< min_begin, tao::pegtl::not_at<tao::pegtl::string<';'>>, tao::pegtl::star<opt_whitespace, objective_term>, opt_whitespace, tao::pegtl::string<';'>, opt_whitespace, tao::pegtl::eol> {};

        struct inequality_type : tao::pegtl::sor<tao::pegtl::string<'<','='>, tao::pegtl::string<'>','='>, tao::pegtl::string<'='>> {};

        struct inequality_identifier : tao::pegtl::seq<
        tao::pegtl::plus<tao::pegtl::sor<tao::pegtl::alnum, tao::pegtl::string<'_'>, tao::pegtl::string<'-'>, tao::pegtl::string<'/'>, tao::pegtl::string<'('>, tao::pegtl::string<')'>, tao::pegtl::string<'{'>, tao::pegtl::string<'}'>, tao::pegtl::string<','> > >
        > {};

        struct new_inequality : tao::pegtl::seq<opt_whitespace, tao::pegtl::not_at<tao::pegtl::string<';'>>, opt_whitespace> {};

        struct inequality_coefficient : tao::pegtl::seq<tao::pegtl::digit, tao::pegtl::star<tao::pegtl::digit>> {};
        struct inequality_variable : variable_name {};
        struct inequality_term : tao::pegtl::seq< tao::pegtl::opt<sign, opt_whitespace>, tao::pegtl::opt<inequality_coefficient, opt_whitespace, tao::pegtl::opt<tao::pegtl::string<'*'>>, opt_whitespace>, inequality_variable> {};
        struct right_hand_side : tao::pegtl::seq< tao::pegtl::opt<sign>, tao::pegtl::digit, tao::pegtl::star<tao::pegtl::digit> > {};

        struct inequality_line : tao::pegtl::seq< new_inequality,
            tao::pegtl::star<opt_whitespace, inequality_term, opt_whitespace, tao::pegtl::opt<tao::pegtl::eol>>,
            opt_whitespace, inequality_type, opt_whitespace, right_hand_side, opt_whitespace, tao::pegtl::string<';'>, tao::pegtl::eol> {};

        struct grammar : tao::pegtl::seq<
                         tao::pegtl::star<comment_line>,
                         objective_line,
                         tao::pegtl::star<inequality_line>,
                         tao::pegtl::until<tao::pegtl::eof>> {};

        template< typename Rule >
            struct action
            : tao::pegtl::nothing< Rule > {};

        struct tmp_storage {
            double objective_coeff = 1.0;
            int constraint_coeff = 1;
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
                    i.add_to_constraint(tmp.constraint_coeff, var);
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
            tao::pegtl::string_input input(input_string, "OPB input");

            if(!tao::pegtl::parse<grammar, action>(input, ilp, tmp))
                throw std::runtime_error("could not read input:\n" + input_string);
            return ilp;
        }

    } 

}

