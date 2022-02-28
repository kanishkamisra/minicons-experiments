library(tidyverse)

fs::dir_ls("data/results", regexp = "*_anli.csv") %>%
  map_df(read_csv) %>%
  mutate(model = fct_reorder(model, accuracy)) %>%
  ggplot(aes(model, accuracy)) +
  geom_col() +
  facet_wrap(~split)
