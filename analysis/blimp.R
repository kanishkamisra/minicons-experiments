library(tidyverse)

blimp <- bind_rows(read_csv("data/results/blimp_multiberts_results_012.csv"), read_csv("data/results/blimp_multiberts_results_34.csv"))%>%
  mutate(
    topic = case_when(
      phenomena == 'animate_subject_passive' | phenomena == 'animate_subject_trans' ~ "argument_structure",
      TRUE ~ topic
    )
  ) %>%
  mutate(
    topic = case_when(
      topic == "determiner_noun_agreement" ~ "det._noun_agreement",
      TRUE ~ topic
    ),
    topic = str_to_title(str_replace_all(topic, "_", " ")),
    topic = case_when(
      topic == "Npi Licensing" ~ "NPI Licensing",
      TRUE ~ topic
    )
  )

blimp %>% count(seed, step, phenomena) %>% View()

bert_blimp <- bind_rows(
  read_csv("data/results/blimp_bert-base-uncased_results.csv") %>% mutate(model = "bert-base"),
  read_csv("data/results/blimp_bert-large-uncased_results.csv") %>% mutate(model = "bert-large")
) %>%
  mutate(
    topic = case_when(
      phenomena == 'animate_subject_passive' | phenomena == 'animate_subject_trans' ~ "argument_structure",
      TRUE ~ topic
    )
  ) %>%
  mutate(
    topic = case_when(
      topic == "determiner_noun_agreement" ~ "det._noun_agreement",
      TRUE ~ topic
    ),
    topic = str_to_title(str_replace_all(topic, "_", " ")),
    topic = case_when(
      topic == "Npi Licensing" ~ "NPI Licensing",
      TRUE ~ topic
    )
  )

bert_results <- bert_blimp %>%
  group_by(model, topic) %>%
  summarize(
    accuracy = mean(good > bad)
  ) %>%
  ungroup() %>%
  filter(model == "bert-base")

bert_results

# COLOR = "#22577E"
COLOR = "#DE834D"

overall <- blimp %>%
  group_by(seed, step) %>%
  summarize(
    accuracy = mean(good > bad)
  ) %>%
  ungroup() %>%
  mutate(
    step = step/10000
  ) %>%
  group_by(step) %>%
  summarize(
    acc = mean(accuracy),
    ste = 1.96 * plotrix::std.error(accuracy),
    acc_high = acc + ste,
    acc_low = acc - ste
  )


p <- blimp %>%
  group_by(field, topic, seed, step) %>%
  summarize(
    accuracy = mean(good > bad)
  ) %>%
  ungroup() %>%
  mutate(
    step = step/10000
  ) %>%
  group_by(topic, step) %>%
  summarize(
    acc = mean(accuracy),
    ste = 1.96 * plotrix::std.error(accuracy),
    acc_high = acc + ste,
    acc_low = acc - ste
  ) %>% 
  ggplot(aes(step, acc)) +
  geom_ribbon(aes(ymin = acc_low, ymax = acc_high), alpha = 0.3, fill = COLOR) +
  # geom_line(data = overall, aes(step, acc), color = "blue") +
  # geom_ribbon(data = overall, aes(x = step, y = acc, ymin=acc_low, ymax=acc_high), alpha = 0.3, fill = "blue") +
  # # geom_point(color = COLOR) +
  geom_line(size = 0.8, color = COLOR) +
  geom_hline(data = bert_results, aes(yintercept = accuracy, linetype = model), show.legend = FALSE) +
  scale_linetype_manual(values=c("dashed", "dotdash")) +
  facet_wrap(~topic) + 
  theme_bw(base_size = 16, base_family = "CMU Serif") +
  theme(
    strip.text = element_text(family = "CMU Sans Serif", size = 11),
    axis.title = element_text(family = "CMU Sans Serif"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Steps (in 10k)",
    y = "Accuracy (95% CI)"
  )

p

ggsave("analysis/multiberts_blimp_2.pdf", p, height = 6, width = 8, device = cairo_pdf, dpi = 300)
