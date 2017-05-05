# pyhmm
python HMM分词程序

### 语料库
>
>1998年1月人民日报语料库
>
>
## train
>
>from api.hmm import HMM
>
>hmm = HMM(model='model/model')
>
>hmm.train(tag_data_path)
>
## test
>
>from api.hmm import HMM
>
>hmm = HMM(model='model/model')
>
>hmm.test('要继续加强追逃追赃等务实合作以及社会领域交流互鉴')
>
>返回分词列表: ['要', '继续', '加强', '追逃', '追赃等务实', '合作', '以及', '社会', '领域', '交流', '互鉴']
>