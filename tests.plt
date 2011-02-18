# show error
plot './errors.log' using 1:2 title 'Errors' with lines 1
# show result
plot [x = -3.14:3.14] (sin(x) + cos(x)) / 2 smooth csplines with lines 2,	\
     '< paste ./test_in.log ./test_out.log' using 1:2 smooth csplines title 'ANN' with lines 1
