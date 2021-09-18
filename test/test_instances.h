#pragma once

namespace LPMP {

const char * matching_3x3 = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
+ 1 x_11 + 1 x_12 + 1 x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
- x_13 - x_23 - x_33 = -1
End)";

const char * covering_problem_3x3 = 
R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
x1 + x2 + x4 >= 1
x1 + x3 + x5 >= 1
x2 + x3 + x6 >= 1
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

const char * covering_problem_2_3x3 = 
R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
-x1 - x2 - x4 <= -1
-x1 - x3 - x5 <= -1
-x2 - x3 - x6 <= -1
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

}
