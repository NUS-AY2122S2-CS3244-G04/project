library(tidyverse)
library(tokenizers)

real_and_fake_dataset = read.csv("Real_and_Fake_Dataset.csv", header = TRUE)

# tokenize by word stems
real_and_fake_dataset$tokenized_text = tokenize_word_stems(real_and_fake_dataset$text)

real_and_fake_tokenized = real_and_fake_dataset %>%
  select(tokenized_text, is_fake)

real_and_fake_tokenized$tokenized_text = sapply(real_and_fake_tokenized$tokenized_text, toString)

write.csv(real_and_fake_tokenized, file=gzfile("Real_and_Fake_Tokenized_Dataset.csv.gz"), row.names = FALSE)
