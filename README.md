# chinese_law_litigant_extract
The extraction of litigant of legal documents based on Bert.(Based on bert-base)


## Install dependencies
You need to install dependencies:
* bert-base
* tensorflow 1.14


## Run
``./run.sh``

## Test(one epochs result)
```bash
accuracy:  99.48%; precision:  93.72%; recall:  97.71%; FB1:  95.67
              DEF: precision:  94.98%; recall:  98.29%; FB1:  96.61  2851
              OTH: precision:   0.00%; recall:   0.00%; FB1:   0.00  1
              PLA: precision:  91.71%; recall:  98.15%; FB1:  94.82  1737
```

## Predict
``python terminal_ner_predict.py``
```text
input: 凃丽波、武汉市鑫吉源商贸有限公司、秦伟毅、武汉福鑫盛物资有限公司、胡正雄：本院刊登于2019年12月31日人民日报G57版的原告付凡华诉你借款合同纠纷一案，已向你送达过开庭公告，法定开庭时间为2020年3月18日9时。因疫情影响，现将开庭时间更正为“自公告发出之日起经过60日即视为送达，并定于送达后第3日上午9时（遇法定节假日顺延）在本院第20号法庭开庭审理，逾期将依法缺席裁判。”特此更正。

output:
DEF, [UNK]丽波, 武汉市鑫吉源商贸有限公司, 秦伟毅, 武汉福鑫盛物资有限公司, 胡正雄
PLA, 付凡华
OTH
```

## To Do List

- [x] Train module
- [ ] dataset public


## Improve
* [UNK]: Chinese rare words, need after-treatment OR add bert vocab.txt

