~/repos/cutechess/build/cutechess-cli \
   -engine conf=sunfish \
   -engine conf=sunfish-king name=sunfish-king arg=`pwd`/models/nnue1.pickle \
   -each tc=3:0+2  \
   -tournament round-robin \
   -rounds 40 \
   -games 1 \
   -concurrency 16 \
   -pgnout qs.pgn \
   -ratinginterval 10 \
   -outcomeinterval 10 \
   -recover \
   -openings file=tests/gaviota-starters.pgn \
   #-debug

   #-engine conf=sunfish-nnue name=er5 stderr=19.err option.EVAL_ROUGHNESS=5 \
   #-engine conf=sunfish-nnue name=er10 stderr=19.err option.EVAL_ROUGHNESS=10 \
   #-engine conf=sunfish-nnue name=er15 stderr=19.err option.EVAL_ROUGHNESS=15 \
   #-engine conf=sunfish-nnue name=er20 stderr=19.err option.EVAL_ROUGHNESS=20 \
   #-engine conf=sunfish-nnue name=er25 stderr=19.err option.EVAL_ROUGHNESS=25 \
