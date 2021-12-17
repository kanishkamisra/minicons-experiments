library(tidyverse)

blimp <- bind_rows(read_csv("data/results/blimp_multiberts_results_012.csv"), read_csv("data/results/blimp_multiberts_results_34.csv"))%>%
  mutate(
    topic = case_when(
      phenomena == 'animate_subject_passive' | phenomena == 'animate_subject_trans' ~ "argument_structure",
      TRUE ~ topic
    )
  )

blimp %>% count(seed, step, phenomena) %>% View()


COLOR = "#22577E"

blimp %>%
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
  mutate(
    topic = case_when(
      topic == "determiner_noun_agreement" ~ "det._noun_agreement",
      TRUE ~ topic
    ),
    topic = str_to_title(str_replace_all(topic, "_", " "))
  ) %>%
  ggplot(aes(step, acc)) +
  geom_ribbon(aes(ymin = acc_low, ymax = acc_high), alpha = 0.4, fill = COLOR) +
  # geom_point(color = COLOR) +
  geom_line(size = 0.8, color = COLOR) +
  facet_wrap(~topic) + 
  theme_bw(base_size = 16, base_family = "CMU Serif") +
  theme(
    strip.text = element_text(family = "CMU Sans Serif Medium", size = 11, face = "bold"),
    axis.title = element_text(family = "CMU Sans Serif Medium")
  ) +
  labs(
    x = "Steps (in 10k)",
    y = "Accuracy (95% CI)"
  )
