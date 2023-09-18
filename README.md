# Log-based Testing
This repository is the basic implementation of our test oracle generation approach **[LogOracle](https://github.com/AnonymousUserOrange/AnonymousRepo#logoracle)** and test case prioritization approach **[LogTCP](https://github.com/AnonymousUserOrange/AnonymousRepo#logtcp)**.


## LogOracle

- [Description](https://github.com/AnonymousUserOrange/AnonymousRepo#description)
- [Structure](https://github.com/AnonymousUserOrange/AnonymousRepo#structure)
- [Datasets](https://github.com/AnonymousUserOrange/AnonymousRepo#datasets&results)
- [Environment](https://github.com/AnonymousUserOrange/AnonymousRepo#environment)
- [Reproducibility](https://github.com/AnonymousUserOrange/AnonymousRepo#reproducibility) 


### Description
*LogOracle* is a general log-based test oracle generation framework, which aims to mine test logs produced during test execution to more sufficiently reflect test behaviors, thereby generate test oracle. Specifically, it includes three key components: log representation, function call representation and test oracle generation. 

### Structure

The project is structured as follows:

```
LogOracle
.
├─ Codet5/      # Scripts of our LogOracle model structrue defination and dataloader for processing raw log data.
├─ config/      # The configuration data.
├─ dataset/     # Input data.
├─ model/       # The directory to save trained model.
├─ utils/      
└─ run.py       # Main entrance code.
```
### Datasets&Results

We use 3 popular open-source Java subjects which have been widely used in many existing studies and results for all the oracles generated are shown in the following table corresponding to the submission.

| ID | Subject | Precision | Recall | F1-score |
| ------- | ------- | :-------: | :-------: | :-------: |
| 1 | ActiveMQ  | 1.00 | 0.96 | 0.98 |
| 2 | Dubbo     | 0.95 | 0.76 | 0.85 |
| 3 | RocketMQ  | 0.95 | 0.83 | 0.89 |

To access and download the subjects' data and the trained model, please follow the instructions below:

1. **Download data:** download processed log file and the trained model from [zenodo](https://zenodo.org/record/8354228).
2. **Unzip:** unzip the compressed file and place them in the root directory `./` and you will get directory `dataset` and `model` which contains the subjects' data and trained model of each subject.


To help better replicate the experiment, we provide sample input datas for all the subjects. 
In the directory `./dataset/process/<module>`, we provide instance files after pre-processing of log producing during test execution for the training, evaluation and testing process.
And in the directory `./model/<module>`, we provide trained model files and you can skip the *training* phase and *test* the oracle generated directly.


### Environment

Key Packages:

- torch
- numpy
- sklearn

### Reproducibility

1. Prepare the data in the directory `./dataset/process/`, including using [LEAM](https://github.com/tianzhaotju/LEAM) to inject mutants into subjects to be tested and collecting logs producing during test executing.
The log processing scripts is in the file `./Codet5/dataloader.py`.

2. Execute the `run.py` script with the following command to train the model:

```
python run.py --phase train --module <module>
```
Example: `python run.py --phase train --module rocketmq/acl`
- The arguments `<phase>` indicate whether to train or test the model.
- The arguments `<module>` indicate the experimental subject.

Notice: You can use the model trained in the [Repo]() and skip this step.

3. Execute the `run.py` script with the following command to generate test oracles and evaluate the *precision, recall and F1-score* of oracles:

```
python run.py --phase test --module <module>
```
Example: `python run.py --phase test --module rocketmq/acl`




## LogTCP

- [Introduction](https://github.com/AnonymousUserOrange/AnonymousRepo#introduction)
- [Project Structure](https://github.com/AnonymousUserOrange/AnonymousRepo#project-structure)
  - [Sample Input Data](https://github.com/AnonymousUserOrange/AnonymousRepo#sample-input-data)

- [Environment](https://github.com/AnonymousUserOrange/AnonymousRepo#environment)
- [Experiment Replication](https://github.com/AnonymousUserOrange/AnonymousRepo#experiment-replication) 
- [Randomness](https://github.com/AnonymousUserOrange/AnonymousRepo#randomness)


### Introduction
*LogTCP* is a general log-based test case prioritization framework, which aims to mine test logs produced during test execution to more sufficiently reflect test behaviors, thereby improving the effectiveness of TCP. Specifically, it includes three key components: log pre-processing, log representation, and test case prioritization. 

### Project Structure

The project is structured as follows:

```
LogTCP
.
├─ data/        # Scripts for the generation of log representation, e.g. the natural language pre-processing on log events, three strategies for log representation. 
├─ dataset/   
│  ├─ input/    # Input data.
│  ├─ process/  # Raw output data. This folder would be generated during the experiments execution.
├─ mutant/      # Scripts for mutant faults selection.
├─ prioritize/  # Scripts for test case prioritization strategies. 
├─ util/      
└─ pipeline.py  # Main entrance code.
```

#### Sample Input Data

To help better replicate the experiment, we provide sample input datas for **10** subjects. In the directory `./dataset/input/`, many files with information about log parsing and mutant faults are placed.

```
./dataset/input/<project>/<module>
├─ drain_results/                # The log parsing results via Drain3.
│  ├─ all_line_event.txt         # The log event IDs for all logs.
│  ├─ log_events.txt             # The log events after log parsing.
├─ config.ini                    # The configuration data.
├─ mutant_id_2_test_case_id.txt  # The relationship between mutant faults and test cases.
├─ test_case.txt                 # All test cases in this module under test.
└─ test_case_event_list.txt      # The log-event sequence for each test case.
```

For ease of understanding, we explain each file in the directory  `./dataset/input/<project>/<module>`:

- `config.ini`: The configuration data,  i.e.,  the hyper-parameters in *FastText* and *Drain3* algorithms.
- `drain_results`: This folder is generated by log parsing tool *Drain3*.
  - `drain_results/all_line_event.txt`: The log event IDs for all logs entered in *Drain3*. It is used to collect the corpus to construct word vectors for the semantics-based log representation.
  - `drain_results/log_events.txt`: The log events parsed by *Drain3*. The log event IDs are assigned in the order of the log events in this file, starting from `0`.
- `test_case.txt`: All test cases in this module under test, including the test case without test execution logs. To assign test case ID, they were sorted in alphabetical order in advance. Likewise, the test case IDs start from `0`.
- `mutant_id_2_test_case_id.txt`: The `mutations.xml` file generated by the mutation tool *PIT* is too large, so we extract the relationship between mutant faults and test cases, and then obtain this file.  For each line, the first number is the mutant fault ID, and the other numbers are the test case IDs that killed it. 
- `test_case_2_event_sequence.txt`: The log-event sequence for each test case. For each line, the first token is the name of test case, and the others are the log event IDs for the test execution logs. 

Due to the general framework of our approach, as long as above information is constructed into the files in the `./dataset/input/<project>/<module>` folder according to the corresponding format, it is easy to apply LogTCP on such subjects. 

### Environment

Key Packages:

- fasttext
- numpy
- sklearn

### Experiment Replication

1. Prepare the data in the directory `./dataset/input/` according to the description [here](https://github.com/AnonymousUserOrange/AnonymousRepo#sample-input-data).

2. Execute the `prioritize.py` script with the following command:

```bash
python pipeline.py --project <project> --module <module> --logs_representation <logs_representation> --prioritization <prioritization> --distance_option <distance_option>
```
Example: `python pipeline.py --project shiro --module core --logs_representation semantics --prioritization arp --distance_option e`
- The arguments `<project>` and `<module>` indicate the experimental subject.
- As for the argument  `<logs_representation>` , the possible values are `count`, `ordering`, and `semantics`, which indicate the count-based,  the ordering-based, and the semantics-based log representation, respectively.
- As for the argument  `<prioritization>` , the possible values are `arp`, `total`,  `additional` and `ideal`, which indicate the adaptive random prioritization strategy, the total strategy, the additional strategy, and the ideal strategy, respectively. It is worth noting that since the semantic-based representation strategy does not involve the concept of coverage, we cannot set  `<prioritization>`  to `total` or `additional`, but set  `<logs_representation>` to `semantics` at the same time.
- The argument `distance_option` indicates the distance calculation method between two log vectors, and it only takes effect when the argument  `<prioritization>` is set to `arp`. The possible values are `e`, `c`,  and `m`, which indicate the Euclidean distance, the Manhattan distance, and the Cosine distance, respectively. 


3. View the prioritization results stored in the folder `./dataset/process/<project>/<module>/priortize_results/`.

### Randomness

We provide our prioritization results for the subject `Shiro-core` as an example in the  `./dataset/process/shiro/core/example_prioritize_results/` [folder](https://github.com/AnonymousUserOrange/AnonymousRepo/tree/main/dataset/process/shiro/core/example_prioritize_results) while the prioritization results for other subjects are shown in the following picture corresponding to the submission.

![avatar](https://github.com/AnonymousUserOrange/AnonymousRepo/blob/main/Effectiveness.png)

You may not get the same results while reproducing the experiments due to the randomness which stems from the following points:

- The word vectors trained by the *FastText* algorithm are not exactly the same.
- The mutation faults used in our experiments are selected randomly.
- In the process of prioritization, when we select a test case from the unselected test cases, there may exist many test cases that meet the current conditions (e.g., the same distance in the adaptive random prioritization strategy, the same coverage in the total and additional strategies and the generation of the ideal prioritization results, and the test cases without test execution logs that need to be appended to the test cases with test execution logs), which leads to the randomness of prioritization results.

To reduce the influence of randomness, we repeated all the TCP techniques involving randomness 5 times and calculated the average results in our study. That's why the results shown in our paper are different from the results in the `example_prioritize_results` folder. 