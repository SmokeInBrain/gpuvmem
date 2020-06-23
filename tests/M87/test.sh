function valid () {
  if [ $? -eq 0 ]; then
    echo OK
  else
    echo ERROR
    exit 1
  fi
}

test=$($1 -i $2/SR1_M87_2017_101_hi_hops_netcal_StokesI.selfcal.LLRR.ms -o $2/residuals.ms -O $2/mod_out.fits -m $2/mod_in_0.fits -I $2/input.dat -p $2/mem/ -X 16 -Y 16 -V 256 --verbose --print-images -z 0.001 -Z 5e-6,5e-5 -t 500000000)
valid $test

#Comment the following lines to see the results of the test
rm -rf $2/residuals.ms
rm -rf $2/mem/
rm $2/alpha.fits
rm $2/mod_out.fits
