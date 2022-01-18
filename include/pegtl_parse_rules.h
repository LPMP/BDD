#pragma once

#include <tao/pegtl.hpp>

// elementary grammar rules for PEGTL

namespace LPMP {

namespace parsing {

    namespace double_ {
        struct plus_minus : tao::pegtl::opt< tao::pegtl::one< '+', '-' > > {};
        struct dot : tao::pegtl::one< '.' > {};

        struct inf : tao::pegtl::seq< tao::pegtl::istring< 'i', 'n', 'f' >,
        tao::pegtl::opt< tao::pegtl::istring< 'i', 'n', 'i', 't', 'y' > > > {};

        struct nan : tao::pegtl::seq< tao::pegtl::istring< 'n', 'a', 'n' >,
        tao::pegtl::opt< tao::pegtl::one< '(' >,
        tao::pegtl::plus< tao::pegtl::alnum >,
        tao::pegtl::one< ')' > > > {};

        template< typename D >
            struct number : tao::pegtl::if_then_else< dot,
            tao::pegtl::plus< D >,
            tao::pegtl::seq< tao::pegtl::plus< D >, tao::pegtl::opt< dot, tao::pegtl::star< D > > > > {};

        struct e : tao::pegtl::one< 'e', 'E' > {};
        struct p : tao::pegtl::one< 'p', 'P' > {};
        struct exponent : tao::pegtl::seq< plus_minus, tao::pegtl::plus< tao::pegtl::digit > > {};

        struct decimal : tao::pegtl::seq< number< tao::pegtl::digit >, tao::pegtl::opt< e, exponent > > {};
        struct hexadecimal : tao::pegtl::seq< tao::pegtl::one< '0' >, tao::pegtl::one< 'x', 'X' >, number< tao::pegtl::xdigit >, tao::pegtl::opt< p, exponent > > {};

        struct grammar : tao::pegtl::seq< plus_minus, tao::pegtl::sor< hexadecimal, decimal, inf, nan > > {};

    }

    struct mand_whitespace : tao::pegtl::plus< tao::pegtl::blank > {}; 
    struct opt_whitespace : tao::pegtl::star< tao::pegtl::blank > {}; 
    struct opt_invisible : tao::pegtl::star< tao::pegtl::sor< tao::pegtl::blank, tao::pegtl::eol > > {};
    struct mand_invisible : tao::pegtl::plus< tao::pegtl::sor< tao::pegtl::blank, tao::pegtl::eol > > {};
    struct positive_integer : tao::pegtl::plus< tao::pegtl::digit > {};

    struct real_number : double_::grammar {};

    struct vector : tao::pegtl::seq< tao::pegtl::string<'['>, opt_whitespace, real_number, tao::pegtl::star< tao::pegtl::seq< mand_whitespace, real_number > >, opt_whitespace, tao::pegtl::string<']'> > {};

} // namespace parsing

} // namespace LPMP

// TODO: test the above definitions with unit testing whether they accept and reject what they are supposed to
