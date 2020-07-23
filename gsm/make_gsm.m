% This script compiles the GSM .c source file to a matlab executable.
% It is written for MinGW v6.3 (Windows) and GCC v6.3.x (Linux). To use with
% other compilers, the compilation command may need to be tuned.

% <quad_precision>
% If set to true, all calculations are performed with quad-precision 128bit floating
% point, using the C <quadmath.h> library. This yields very accurate results (probably
% to within 1 ulp in double precision) but may incur up to 40-fold slowdown. 
% Normally this is only used for accuracy evaluation.
quad_precision = false;

% <fast_math> 
% If set to true, the '-Ofast' optimization flag is used, which yields a moderate speedup.
% If false, the more conservative '-O2' flag is used, and intermediate results are forced
% to be stored in the memory rather than in CPU registers (which may be 80-bit rather
% than 64-bit). This makes the arithmetic operations compatible with the IEEE 754 standard
% and thus more consistent across different computers.
% Note that setting this flag to false does not in itself guarantee cross-system 
% consistency, as slightly different results can also be due to different implementations
% of <math.h> functions used in the algorithm, such as exp, log etc.
% Unless consistency across different computers is highly important, it is recommended
% to use fast_math=true.
% Note: This flag is irrelevant if <quad_precision> = true
fast_math = true;

source_filename = 'gsm_v5_1_mex.c';

fprintf('Compiling source file ''%s''...\n', source_filename);


%% Construct mex arguments
args = cell(0,1);

args = [args; {'-DNDEBUG'}];

if quad_precision
    args = [args; {'-Dgsm_intermediate_numerical_type=4'}];
    args = [args; {'LINKLIBS="$LINKLIBS -lquadmath"'}]; 
    args = [args; {'COPTIMFLAGS="-O2 -fwrapv"'}];
elseif fast_math
    args = [args; {'-Dgsm_intermediate_numerical_type=2'}];
    args = [args; {'COPTIMFLAGS="-Ofast -mfpmath=sse -fwrapv"'}];
else
    args = [args; {'-Dgsm_intermediate_numerical_type=2'}];
    args = [args; {'COPTIMFLAGS="-O2 -ffloat-store -fwrapv"'}];
end

args = [args; {'CDEBUGFLAGS=""'}];
args = [args; {'LDDEBUGFLAGS=""'}];
args = [args; {'DEBUGFLAGS=""'}];
args = [args; {'LINKDEBUGFLAGS=""'}];
args = [args; {'CFLAGS="-fPIC -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -Wunsuffixed-float-constants"'}];

args = [args; {source_filename}];

%% Construct mex command
mexcmd = sprintf('mex_result = mex(args{1}');

for i=2:numel(args)
    mexcmd = sprintf('%s, args{%d}', mexcmd, i);
end

mexcmd = [mexcmd, ');'];

% Execute mex command
eval(mexcmd);
