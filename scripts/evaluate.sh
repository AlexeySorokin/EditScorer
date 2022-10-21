#!/bin/bash
cd $(dirname $0)/..

while getopts "i:r:" opt
  do
      case $opt in
      i) INFILE=$OPTARG ;;
      r) RESULT_FILE=$OPTARG ;;
      *) echo "Unknown option $opt"
        exit 1 ;;
      esac
  done

ORIG_FILE=${INFILE}".tmp"
grep -P "^S " ${INFILE} > ${ORIG_FILE}
sed -i "s/^S //" ${ORIG_FILE}
M2_OUTFILE=${RESULT_FILE}".m2"
COMPARISON_OUTFILE=${RESULT_FILE}".m2log"
errant_parallel -orig $ORIG_FILE -cor $RESULT_FILE -out $M2_OUTFILE
errant_compare -hyp $M2_OUTFILE -ref $INFILE -v -cat 3 > ${COMPARISON_OUTFILE}
tail --lines=+"$(echo $(grep -m 2 -ne "Span-Based Correction" ${COMPARISON_OUTFILE} | tail -n 1) | cut -d : -f 1)" ${COMPARISON_OUTFILE}
rm -r $ORIG_FILE