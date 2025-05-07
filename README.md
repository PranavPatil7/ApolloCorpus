# ApolloCorpus, Multilingual Medical Benchmark for LLMs

<div align=center>
<img src="assets/zoo.png" width = "640" alt="zoo" align=center />
</div>

⚡LLM Zoo is a project that provides data, models, and evaluation benchmark for large language models.⚡ 

## 🤔 Motivation
- Break  "AI supremacy"  and democratize ChatGPT

> "AI supremacy" is understood as a company's absolute leadership and monopoly position in an AI field, which may even include exclusive capabilities beyond general artificial intelligence. This is unacceptable for AI community and may even lead to individual influence on the direction of the human future, thus bringing various hazards to human society.

- Make ChatGPT-like LLM accessible across countries and languages
- Make AI open again. Every person, regardless of their skin color or place of birth, should have equal access to the
  technology gifted by the creator. For example, many pioneers have made great efforts to spread the use of light bulbs
  and vaccines to developing countries. Similarly, ChatGPT, one of the greatest technological advancements in modern
  history, should also be made available to all.

## 🎬 Get started
### Install
Run the following command to install the required packages:

```angular2html
pip install -r requirements.txt
```

### CLI Inference
```bash
python -m llmzoo.deploy.cli --model-path /path/to/weights/
```
For example, for `Phoenix`, run
```bash
python -m llmzoo.deploy.cli --model-path FreedomIntelligence/phoenix-inst-chat-7b
```
and it will download the model from Hugging Face automatically. For `Chimera`, please follow this [instruction](https://github.com/FreedomIntelligence/LLMZoo#chimera-llm-mainly-for-latin-and-cyrillic-languages) to prepare the weights.

> Check [here](#-deployment) for deploying a web application.


## 📚 Data
### Overview
I used the following two types of data for training `Phoenix` and `Chimera`:

<details><summary><b>Instruction data</b></summary>

- Multilingual instructions (language-agnostic instructions with post-translation)

```diff
+ Self-Instructed / Translated (Instruction, Input) in Language A
- ---(Step 1) Translation --->
+ (Instruction, Input) in Language B (B is randomly sampled w.r.t. the probability distribution of realistic languages)
- ---(Step 2) Generate--->
+ Output in Language B
```

- User-centered instructions
```diff
+ (Role, Instruction, Input) seeds
- ---(Step 1) Self Instruct--->
+ (Role, Instruction, Input) samples
- ---(Step 2) generate output Instruct--->
+ (Role, Instruction, Input) ---> Output
```

</details>

<details><summary><b>Conversation data</b></summary>

- User-shared conversations

```diff
+ ChatGPT conversations shared on the Internet
- ---(Step 1) Crawl--->
+ Multi-round conversation data
```
</details>

> Check [InstructionZoo](https://github.com/FreedomIntelligence/InstructionZoo) for the collection of instruction datasets.

> Check [GPT-API-Accelerate Tool](https://github.com/FreedomIntelligence/GPT-API-Accelerate) for faster data generation using ChatGPT.

### Download
- [phoenix-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/phoenix-sft-data-v1): The data used for training Phoenix and Chimera.


## 🐼 Models
### Overview of existing models
| Model                         | Backbone |  #Params | Open-source model | Open-source data | Claimed language | Post-training (instruction) | Post-training (conversation) | Release date |
|-------------------------------|----------|---------:|------------------:|-----------------:|-----------------:|----------------------------:|-----------------------------:|-------------:|
| ChatGPT                       | -        |        - |                 ❌ |                ❌ |            multi |                             |                              |     11/30/22 |
| Wenxin                        | -        |        - |                 ❌ |                ❌ |               zh |                             |                              |     03/16/23 |
| ChatGLM                       | GLM      |       6B |                 ✅ |                ❌ |           en, zh |                             |                              |     03/16/23 |
| Alpaca                        | LLaMA    |       7B |                 ✅ |                ✅ |               en |                     52K, en |                            ❌ |     03/13/23 |
| Dolly                         | GPT-J    |       6B |                 ✅ |                ✅ |               en |                     52K, en |                            ❌ |     03/24/23 |
| BELLE                         | BLOOMZ   |       7B |                 ✅ |                ✅ |               zh |                    1.5M, zh |                            ❌ |     03/26/23 |
| Guanaco                       | LLaMA    |       7B |                 ✅ |                ✅ |   en, zh, ja, de |                 534K, multi |                            ❌ |     03/26/23 |
| Chinese-LLaMA-Alpaca          | LLaMA    |    7/13B |                 ✅ |                ✅ |           en, zh |                2M/3M, en/zh |                            ❌ |     03/28/23 |
| LuoTuo                        | LLaMA    |       7B |                 ✅ |                ✅ |               zh |                     52K, zh |                            ❌ |     03/31/23 |
| Vicuna                        | LLaMA    |    7/13B |                 ✅ |                ✅ |               en |                           ❌ |                   70K, multi |     03/13/23 |
| Koala                         | LLaMA    |      13B |                 ✅ |                ✅ |               en |                    355K, en |                     117K, en |     04/03/23 |
| BAIZE                         | LLaMA    | 7/13/30B |                 ✅ |                ✅ |               en |                          52K, en |                   111.5K, en |     04/04/23 |
| Phoenix (Ours)                | BLOOMZ   |       7B |                 ✅ |                ✅ |            multi |                         40+ |                          40+ |     04/08/23 |
| Latin Phoenix: Chimera (Ours) | LLaMA    |    7/13B |                 ✅ |                ✅ |    multi (Latin) |                       Latin |                        Latin |     04/08/23 |

<details><summary><b>The key difference between existing models and ours.</b></summary>

> The key difference in our models is that we utilize two sets of data, namely **instructions** and **conversations**, which were previously only used by Alpaca and Vicuna respectively. We believe that incorporating both types of data is essential for a recipe  to achieve a proficient language model. The rationale  is that *the **instruction** data helps to tame language  models to adhere to human instructions and fulfill their information requirements*, while *the **conversation** data facilitates the development of conversational skills in the model*. Together, these two types of data complement each other to create a more well-rounded language model.
</details>

### Phoenix (LLM across Languages)
<details><summary><b>The philosophy to name</b></summary>

> The first model is named **Phoenix**. In Chinese culture, the Phoenix is commonly regarded as a symbol of *the king of birds*; as the saying goes "百鸟朝凤", indicating its ability to coordinate with all birds, even if they speak different languages. We refer to Phoenix as the one capable of understanding and speaking hundreds of (bird) languages. More importantly, **Phoenix** is the totem of "the Chinese University of Hong Kong, Shenzhen" (CUHKSZ); it goes without saying this is also for the Chinese University of Hong Kong (CUHK).

</details>

| Model                | Backbone      | Data         | Link                                                                          |
|----------------------|---------------|--------------|-------------------------------------------------------------------------------|
| Phoenix-chat-7b      | BLOOMZ-7b1-mt | Conversation | [parameters](https://huggingface.co/FreedomIntelligence/phoenix-chat-7b)      |
| Phoenix-inst-chat-7b | BLOOMZ-7b1-mt | Instruction + Conversation | [parameters](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b) |
| Phoenix-inst-chat-7b-int4 | BLOOMZ-7b1-mt | Instruction + Conversation | [parameters](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b-int4) |

### Chimera (LLM mainly for Latin and Cyrillic languages)
<details><summary><b>The philosophy to name</b></summary>

> The philosophy to name: The biggest barrier to LLM is that we do not have enough candidate names for LLMs, as LLAMA, Guanaco, Vicuna, and Alpaca have already been used, and there are no more members in the camel family. Therefore, we find a similar hybrid creature in Greek mythology, [Chimera](https://en.wikipedia.org/wiki/Chimera_(mythology)), composed of different Lycia and Asia Minor animal parts. Coincidentally, it is a hero/role in DOTA (and also Warcraft III). It could therefore be used to memorize a period of playing games overnight during high school and undergraduate time.
</details>

| Model                 | Backbone  | Data                       | Link                                                                                         |
|-----------------------|-----------|----------------------------|----------------------------------------------------------------------------------------------|
| Chimera-chat-7b       | LLaMA-7b  | Conversation               | [parameters (delta)](https://huggingface.co/FreedomIntelligence/chimera-chat-7b-delta)       |
| Chimera-chat-13b      | LLaMA-13b | Conversation               | [parameters (delta)](https://huggingface.co/FreedomIntelligence/chimera-chat-13b-delta)      |
| Chimera-inst-chat-7b  | LLaMA-7b  | Instruction + Conversation | [parameters (delta)](https://huggingface.co/FreedomIntelligence/chimera-inst-chat-7b-delta)  |
| Chimera-inst-chat-13b | LLaMA-13b | Instruction + Conversation | [parameters (delta)](https://huggingface.co/FreedomIntelligence/chimera-inst-chat-13b-delta) |

Due to LLaMA's license restrictions, we follow [FastChat](https://github.com/lm-sys/FastChat) to release our delta weights. To use Chimera, download the original [LLaMA weights](https://huggingface.co/docs/transformers/main/model_doc/llama) and run the script:

```bash
python tools/apply_delta.py \
 --base /path/to/llama-13b \
 --target /output/path/to/chimera-inst-chat-13b \
 --delta FreedomIntelligence/chimera-inst-chat-13b-delta
```

### CAMEL (Chinese And Medically Enhanced Langauge models)
<details><summary><b>The philosophy to name</b></summary>

> The philosophy to name: Its Chinese name is [HuatuoGPT](https://github.com/FreedomIntelligence/HuatuoGPT) or 华佗GPT to commemorate the great Chinese physician named Hua Tuo (华佗), who lived around 200 AC. Training is already finished; we will release it in two weeks; some efforts are needed to deploy it in public cloud servers in case of massive requests.
</details>

Check our models in [HuatuoGPT](https://github.com/FreedomIntelligence/HuatuoGPT) or try our [demo](https://www.huatuogpt.cn/) .
Similar biomedical models could be seen in [biomedical LLMs](assets/biomedical-models.md).

<details><summary><b>More models in the future</b></summary>

### Legal GPT (coming soon)

### Vision-Language Models (coming soon)

### Retrieval-augmented Models (coming soon)

</details>

## 🧐 Evaluation and Benchmark

I provide a bilingual, multidimensional comparison across different open-source models with ours.

### Chinese

* Automatic Evaluation Using GPT-4:

| Model                                           | Ratio   |
|-------------------------------------------------|---------|
| Phoenix-inst-chat-7b vs. **ChatGPT**            | 85.2\%  |
| Phoenix-inst-chat-7b vs. **ChatGLM-6b**         | 94.6\%  |
| Phoenix-inst-chat-7b vs. **Baidu-Wenxin**       | 96.8\%  |
| **Phoenix-inst-chat-7b** vs. MOSS-moon-003-sft  | 109.7\% |
| **Phoenix-inst-chat-7b** vs. BELLE-7b-2m        | 122.7\% |
| **Phoenix-inst-chat-7b** vs. Chinese-Alpaca-7b  | 135.3\% |
| **Phoenix-inst-chat-7b** vs. Chinese-Alpaca-13b | 125.2\% |

**Observation**: It shows that Phoenix-chat-7b achieves 85.2\% performance of ChatGPT in Chinese. It slightly underperforms Baidu-Wenxin (96.8\%) and ChatGLM-6b (94.6 \%), both are not fully open-source;  ChatGLM-6b only provides model weights without training data and details. Although Phoenix is a multilingual LLM, it achieves SOTA performance among all open-source Chinese LLMs.

* Human Evaluation:

|                                      | win | tie | lose  |
|--------------------------------------|:---:|:---:|:-----:|
| Phoenix vs. **ChatGPT**              | 12  |  35 |  53   |
| Phoenix vs. **ChatGLM-6b**           | 36  |  11 |  53   |
| Phoenix vs. **Baidu-Wenxin**         | 29  |  25 |  46   |
| **Phoenix** vs. BELLE-7b-2m          | 55  |  31 |  14   |
| **Phoenix** vs. Chinese-Alpaca-13b   | 56  |  31 |  13   |

**Observation**: It shows that the human evaluation results show the same trend as the automatic evaluation results.

### English

* Automatic Evaluation Using GPT-4:

| Model                                  | Ratio      |
|----------------------------------------|------------|
| Chimera-chat-7b vs.  **ChatGPT**       | 85.2\%     |
| Chimera-chat-13b vs.  **ChatGPT**      | 92.6\%     |
| Chimera-inst-chat-13b vs.  **ChatGPT** | **96.6\%** |




## 🏭 Deployment

### Launch a controller
```shell
python -m llmzoo.deploy.webapp.controller
```

### Launch a model worker
```shell
python -m llmzoo.deploy.webapp.model_worker --model-path /path/to/weights/
```

### Launch a gradio web server
```shell
python -m llmzoo.deploy.webapp.gradio_web_server
```
Now, you can open your browser and chat with a model.


## 😀 Training by yourself
### Prepare the data
You can either download the [phoenix-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/phoenix-sft-data-v1) data or prepare your own data. Put your data on the path `data/data.json`.

### Training
For `Phoenix`, run
```shell
bash scripts/train_phoenix_7b.sh
```
For `Chimera`, prepare the LLaMA weights following [this instruction](https://huggingface.co/docs/transformers/main/model_doc/llama) and run
```shell
bash scripts/train_chimera_7b.sh
bash scripts/train_chimera_13b.sh
```

## 🤖 Limitations

My goal in releasing My models is to assist our community in better replicating ChatGPT/GPT4. I am not targeting
competition with other competitors, as benchmarking models is a challenging task. My models face similar models to
those of ChatGPT/GPT4, which include:

- Lack of common sense: our models may not always have the ability to apply common sense knowledge to situations, which
  can lead to nonsensical or inappropriate responses.

- Limited knowledge domain: our models' knowledge is based on the data it was trained on, and it may not have the
  ability to provide accurate or relevant responses outside that domain.

- Biases: our models may have biases that reflect the biases in the data it was trained on, which can result in
  unintended consequences or unfair treatment.

- Inability to understand emotions: While our models can understand language, it may not always be able to understand
  the emotional tone behind it, which can lead to inappropriate or insensitive responses.

- Misunderstandings due to context: our models may misunderstand the context of a conversation, leading to
  misinterpretation and incorrect responses.



[![Star History Chart](https://api.star-history.com/svg?repos=FreedomIntelligence/LLMZoo&type=Date)](https://star-history.com/#FreedomIntelligence/LLMZoo&Date)
