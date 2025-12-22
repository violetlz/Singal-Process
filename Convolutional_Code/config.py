# config.py

CONFIG = {

    "GLOBAL": {
        "use_gpu": True
    },

    # 卷积码配置
    "CONV": {
        # generator polynomials (binary form)
        # example: (7,5)_oct = [[1,1,1],[1,0,1]]
        "polynomials": [
            [1, 1, 1],
            [1, 0, 1]
        ],

        "constraint_len": 3,

        # zero-tail
        "flush": True,

        # Viterbi
        "viterbi": {
            "metric": "hamming",   # hamming / euclidean
            "traceback": None      # None = full traceback
        }
    }
}
