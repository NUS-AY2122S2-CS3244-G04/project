library(tidyverse)

buzzfeed_fake = read.csv("BuzzFeed_fake_news_content.csv", header = TRUE)
buzzfeed_real = read.csv("BuzzFeed_real_news_content.csv", header = TRUE)
fakes = read.csv("FA-KES-Dataset.csv", header = TRUE)
fake = read.csv("Fake.csv", header = TRUE)
politifact_real = read.csv("PolitiFact_real_news_content.csv", header = TRUE)
true = read.csv("True.csv", header = TRUE)

# 1 if fake, 0 if real

# buzzfeed fake
buzzfeed_fake$is_fake = rep(1, nrow(buzzfeed_fake))

buzzfeed_fake = buzzfeed_fake %>%
  select(text, is_fake)

# buzzfeed real
buzzfeed_real$is_fake = rep(0, nrow(buzzfeed_real))

buzzfeed_real = buzzfeed_real %>%
  select(text, is_fake)

# fake
fake$is_fake = rep(1, nrow(fake))

fake = fake %>%
  select(text, is_fake)

# fakes
# 'labels' column -> 1 represents real and 0 represents fake
# reversed the labels
fakes$is_fake = ifelse(fakes$labels == 0, 1, 0)

fakes = fakes %>%
  rename(text = article_content) %>%
  select(text, is_fake)

# politifact
politifact_real$is_fake = rep(0, nrow(politifact_real))

politifact_real = politifact_real %>%
  select(text, is_fake)

# true
true$is_fake = rep(0, nrow(true))

true = true %>%
  select(text, is_fake)


# combining all the datasets
real_and_fake_dataset = rbind(buzzfeed_fake, buzzfeed_real, fake, fakes, politifact_real, true)

write.csv(real_and_fake_dataset, file=gzfile("Real_and_Fake_Dataset.csv.gz"), row.names = FALSE)
