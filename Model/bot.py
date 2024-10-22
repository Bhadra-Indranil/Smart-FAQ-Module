  import json
import torch
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer,util
from transformers import T5Tokenizer, TFT5ForConditionalGeneration , T5ForConditionalGeneration , Trainer , TrainingArguments , T5Config
from peft import LoraConfig, get_peft_model, TaskType


def read_json(File_path):
  with open(File_path) as f:
    data = json.load(f)
  data_cat = []

  for cat in data:
    for faq in data[cat]:
      question = faq["question"]
      answer = faq["answer"]
      data_cat.append({"question": question, "target": answer})
  return data_cat , pd.DataFrame(data_cat)

def compute_embedding(data,model):
  questions = [item['question'] for item in data]
  question_embeddings = model.encode(questions, convert_to_tensor=True)
  return question_embeddings


def extend_dataset_paraphrasing(question,tokenizer,model,num_return_seq=10):
  '''
  As we need a diverse dataset , we need to go for paraphrasing . So that our model can learn and be trained in the dataset and be able to generate answer
  '''
  input_text = f"paraphrase : {question} </s>"

  input_ids = tokenizer.encode(input_text, return_tensors="pt",max_length=512,truncation=True)

  output = model.generate(
      input_ids,
      max_length=512,
      num_beams=20,
      num_return_sequences=num_return_seq,
      # temperature=1.5,
      early_stopping=True
  )

  paraphrases = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]
  return paraphrases


def augment_q(df,tokenizer,model):
  augmented_data = []
  for index,row in df.iterrows():
    question = row['question']
    target = row['target']

    paraphrases = extend_dataset_paraphrasing(question,tokenizer,model)
    paraphased_ans = extend_dataset_paraphrasing(target,tokenizer,model)
    for paraphrase,paraphrase_ans in zip(paraphrases,paraphased_ans):
      augmented_data.append({'question': paraphrase, 'target': paraphrase_ans})
  augmented_df = pd.DataFrame(augmented_data)
  return pd.concat([df,augmented_df],ignore_index=True)




def find_similar_question(user_q,data, question_embeddings, model, threshold=0.7):
    
    '''
    This function finds a similar question in the given dataset using the provided user question and the pre-trained SentenceTransformer model.
    The function returns the most similar question from the dataset and its similarity score.
    This function find the similar question using the cosine similarity from the vector matrix'''
    
    user_question_embedding = model.encode(user_q, convert_to_tensor=True)

    
    similarities = util.pytorch_cos_sim(user_question_embedding, question_embeddings)

    
    similarities_cpu = similarities.detach().cpu()  

    
    max_match_id = torch.argmax(similarities_cpu).item()  
    max_similarity = similarities_cpu[0][max_match_id].item()  

    if max_similarity >= threshold and 0 <= max_match_id < len(data):
      return data[max_match_id]['target'], max_similarity

    else:
        return None, max_similarity
    


def generator_model(train_df, eval_df):
    """
    This function fine-tunes a pre-trained T5 model for a question-answering task using a given training and evaluation dataset.#+
    The model is trained on the provided data and saved in the 'fine_tuned_lora_t5' directory.#+

    Parameters:
    train_df (pandas.DataFrame): A DataFrame containing the training data with 'question' and 'target' columns.#+
    eval_df (pandas.DataFrame): A DataFrame containing the evaluation data with 'question' and 'target' columns.#+
#+
    Returns:
    None
    """
    
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    
    train_input_encodings = tokenizer(train_df['question'].tolist(), truncation=True, padding=True, return_tensors="pt")
    train_target_encodings = tokenizer(train_df['target'].tolist(), truncation=True, padding=True, return_tensors="pt")

    
    eval_input_encodings = tokenizer(eval_df['question'].tolist(), truncation=True, padding=True, return_tensors="pt")
    eval_target_encodings = tokenizer(eval_df['target'].tolist(), truncation=True, padding=True, return_tensors="pt")

    
    class QA_Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, targets):
            self.encodings = encodings
            self.targets = targets

        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.targets['input_ids'][idx]
            }

        def __len__(self):
            return len(self.encodings['input_ids'])

    
    train_dataset = QA_Dataset(train_input_encodings, train_target_encodings)
    eval_dataset = QA_Dataset(eval_input_encodings, eval_target_encodings)

    
    training_args = TrainingArguments(
        output_dir="./lora_t5",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=2e-4,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 
        tokenizer=tokenizer
    )

    
    trainer.train()
    model.save_pretrained("fine_tuned_lora_t5")
    tokenizer.save_pretrained("fine_tuned_lora_t5")

def generate_answer(question, tokenizer, model):
    input_text = f"question: {question.strip()}"

    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  
    input_ids = input_ids.to(device)  


    outputs = model.generate(input_ids, max_length=70, num_beams=3, early_stopping=True , min_length=20)

    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer




data,df = read_json('/kaggle/input/faq-json/faqs.json')

paraphrase_model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")
paraphrase_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")

augmented_faq = augment_q(df,paraphrase_tokenizer,paraphrase_model)
augmented_faq.to_csv('augmented_faq.csv',index=False)
augmented_faq = pd.read_csv('augmented_faq.csv')

sbert = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = compute_embedding(data,sbert)


train_df, eval_df = train_test_split(augmented_faq, test_size=0.2, random_state=42)

generator_model(train_df,eval_df)




tokenizer = T5Tokenizer.from_pretrained('fine_tuned_lora_t5')
model = T5ForConditionalGeneration.from_pretrained('fine_tuned_lora_t5')


thresh = 0.6


print("Bot is running... Press [Exit] to quite")
while True:
  user = input("You : ").strip()
  if user == "Exit":
    break
  answer,similarity = find_similar_question(user,data,question_embeddings,sbert,thresh)
  if answer:
    print(f"Bot : {answer}")
  else:
    answer = generate_answer(user,tokenizer,model)
    print(f"Bot : {answer}")
