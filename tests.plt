# show test results
plot [-0.05:1.05][0.8:1.2] './test_in.log' using 1:(1) title 'Samples',	\
     './test_out.log' using (cos($1) + cos($1 + $2)):(sin($1) + sin($1 + $2)) title 'ANN results'
