~/repos/cutechess/build/cutechess-cli \
   -engine conf=sunfish \
   -engine conf=sunfish-nnue name=sun19 arg=model_19.pickle stderr=19.err \
   -engine conf=sunfish-nnue name=sun39 arg=model_39.pickle stderr=39.err \
   -engine conf=sunfish-nnue name=sun59 arg=model_59.pickle stderr=59.err \
   -engine conf=sunfish-nnue name=sun79 arg=model_79.pickle stderr=79.err \
   -engine conf=sunfish-nnue name=sun99 arg=model_99.pickle stderr=99.err \
   -each tc=3:0+2  \
   -tournament round-robin \
   -rounds 40 \
   -games 1 \
   -concurrency 16 \
   -pgnout qs.pgn \
   -ratinginterval 10 \
   -outcomeinterval 10 \
   -recover \
   -openings file=gaviota-starters.pgn \
   #-debug

   #-engine conf=sunfish-nnue name=er5 stderr=19.err option.EVAL_ROUGHNESS=5 \
   #-engine conf=sunfish-nnue name=er10 stderr=19.err option.EVAL_ROUGHNESS=10 \
   #-engine conf=sunfish-nnue name=er15 stderr=19.err option.EVAL_ROUGHNESS=15 \
   #-engine conf=sunfish-nnue name=er20 stderr=19.err option.EVAL_ROUGHNESS=20 \
   #-engine conf=sunfish-nnue name=er25 stderr=19.err option.EVAL_ROUGHNESS=25 \
