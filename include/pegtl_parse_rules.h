#pragma once

#include <tao/pegtl.hpp>

// elementary grammar rules for PEGTL

namespace LPMP {

namespace parsing {

struct mand_whitespace : tao::pegtl::plus< tao::pegtl::blank > {}; 
struct opt_whitespace : tao::pegtl::star< tao::pegtl::blank > {}; 
struct opt_invisible : tao::pegtl::star< tao::pegtl::sor< tao::pegtl::blank, tao::pegtl::eol > > {};
struct mand_invisible : tao::pegtl::plus< tao::pegtl::sor< tao::pegtl::blank, tao::pegtl::eol > > {};
struct positive_integer : tao::pegtl::plus< tao::pegtl::digit > {};

struct real_number_standard : tao::pegtl::sor<
                              tao::pegtl::seq< tao::pegtl::opt< tao::pegtl::one<'+','-'> >, tao::pegtl::plus<tao::pegtl::digit>, tao::pegtl::opt< tao::pegtl::seq< tao::pegtl::string<'.'>, tao::pegtl::star<tao::pegtl::digit> > > >,
                              tao::pegtl::string<'I','n','f'>,
                              tao::pegtl::string<'i','n','f'>
                              > {}; 
struct real_number_smaller1 : tao::pegtl::seq< tao::pegtl::opt< tao::pegtl::one<'+','-'> >, tao::pegtl::string<'.'>, tao::pegtl::plus< tao::pegtl::digit > > {};
struct real_number_exponential : tao::pegtl::seq< tao::pegtl::opt< tao::pegtl::one<'+','-'> >, tao::pegtl::star< tao::pegtl::digit >, tao::pegtl::opt<tao::pegtl::seq< tao::pegtl::string<'.'>, tao::pegtl::star< tao::pegtl::digit>>>, tao::pegtl::string<'e'>, tao::pegtl::opt< tao::pegtl::one<'+','-'> >, tao::pegtl::plus< tao::pegtl::digit > > {};
struct real_number : tao::pegtl::sor<real_number_exponential, real_number_standard, real_number_smaller1> {};

struct vector : tao::pegtl::seq< tao::pegtl::string<'['>, opt_whitespace, real_number, tao::pegtl::star< tao::pegtl::seq< mand_whitespace, real_number > >, opt_whitespace, tao::pegtl::string<']'> > {};

} // namespace parsing

} // namespace LPMP

// TODO: test the above definitions with unit testing whether they accept and reject what they are supposed to
