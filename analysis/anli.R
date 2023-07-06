library(tidyverse)

read_lines("data/anli/dev-labels.lst") %>%
  as.integer() %>%
  table()

random <- c(0.5097911, 0.5158549)

model_meta <- read_csv("data/model_meta.csv")

results <- fs::dir_ls("data/results/anli/", regexp = "*_anli.csv") %>%
  map_df(read_csv) %>%
  mutate(
    model = str_remove(model, "(google_)"),
    model = str_remove(model, "\\-generator"),
  ) 

dev_res <- results %>% filter(split == "dev")
test_res <- results %>% filter(split == "test")

cor.test(dev_res$parameters, dev_res$accuracy)

fit = lm(log10(parameters) ~ accuracy, data = dev_res)

summary(fit)

predict(fit, tibble(accuracy = c(0.58, 0.6, 0.7, 0.8, 0.9)))

p <- results %>%
  inner_join(model_meta) %>%
  mutate(
    split = case_when(
      split == "dev" ~ "Dev",
      split == "test" ~ "Test"
    )
  ) %>%
  ggplot(aes(parameters/1e6, accuracy, color = log(parameters/1e6))) +
  geom_point(size = 3) +
  # geom_smooth(method = "lm") +
  facet_wrap(~split) +
  # ggsci::scale_color_material("deep-purple") +
  scale_color_distiller(palette = "BuPu", direction = 1) +
  # scale_color_identity() +
  scale_y_continuous(breaks = c(0.5, 0.54, 0.58, 0.62)) +
  scale_x_log10() +
  theme_bw(base_size = 16, base_family = "CMU Serif") +
  theme(
    legend.position = "none",
    strip.text = element_text(family = "CMU Sans Serif Medium", size = 11, face = "bold"),
    axis.title = element_text(family = "CMU Sans Serif Medium"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in million)",
    y = "Accuracy"
  )

p

ggsave("analysis/anli.pdf", p, height = 2.7, width = 4.5, device = cairo_pdf, dpi = 300)


p <- results %>%
  inner_join(model_meta) %>%
  mutate(
    split = case_when(
      split == "dev" ~ "Dev",
      split == "test" ~ "Test"
    )
  ) %>%
  filter(split == "Dev") %>%
  ggplot(aes(parameters, accuracy, color = log(parameters/1e6))) +
  geom_point(size = 3, color = "black") +
  geom_smooth() +
  annotate("text", x = 10^(3.2+6), y = 0.51, label = "italic(R) ^ 2 == 0.48", size = 5, color = "black", fontface = "italic", family = "CMU Serif", parse = TRUE) +
  # facet_wrap(~split) +
  # ggsci::scale_color_material("deep-purple") +
  # scale_color_distiller(palette = "BuPu", direction = 1) +
  # scale_color_identity() +
  scale_y_continuous(breaks = c(0.5, 0.54, 0.58, 0.62), limits = c(0.49, 0.63)) +
  scale_x_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)), limit = c(1e7, 1e10)) +
  theme_bw(base_size = 18, base_family = "CMU Serif") +
  theme(
    legend.position = "none",
    strip.text = element_text(family = "CMU Sans Serif Medium", size = 11, face = "bold"),
    axis.title = element_text(family = "CMU Sans Serif Medium"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in million)",
    y = "Accuracy"
  )

p

ggsave("analysis/anli.pdf", height = 4, width = 4.5, device = cairo_pdf, dpi = 300)

fs::dir_ls("data/results/anli/", regexp = "*_anli.csv") %>%
  map_df(read_csv) %>%
  distinct(model, parameters) %>% 
  write_csv("data/meta.csv")

levels = c('A-b',
           'A-l',
           'A-xl',
           'A-xxl',
           'dB-b',
           'B-b',
           'B-l',
           'E-s',
           'E-b',
           'E-l',
           'dR-b',
           'R-b',
           'R-l',
           'dGPT2',
           'GPT',
           'GPT2',
           'GPT2-m',
           'GPT2-l',
           'GPT2-xl',
           'GPT-Neo-125M',
           'GPT-Neo-1.3b',
           'GPT-Neo-2.7B',
           'GPT-J')

labels = c('A-b',
           'A-l',
           'A-xl',
           'A-xxl',
           'dB-b',
           'B-b',
           'B-l',
           'E-s',
           'E-b',
           'E-l',
           'dR-b',
           'R-b',
           'R-l',
           'dGPT2',
           'GPT',
           'GPT2',
           'GPT2-m',
           'GPT2-l',
           'GPT2-xl',
           'Neo-125M',
           'Neo-1.3b',
           'Neo-2.7B',
           'GPT-J')

p <- results %>% 
  inner_join(model_meta) %>%
  filter(split == "dev") %>%
  mutate(shorter = factor(shorter, levels = levels, labels = labels)) %>%
  ggplot(aes(shorter, accuracy, color = color, fill = color)) +
  geom_col() +
  geom_hline(yintercept = 0.5097911, linetype = "dashed", color = "black", size = 1) +
  geom_hline(yintercept = 0.9197, linetype = "dotted", size = 1) +
  geom_hline(yintercept = 0.9290, linetype = "solid") +
  annotate("text", x = 2.5, y = 0.72, label = "ALBERT", size = 5.5, color = "#2e59a8", fontface = 2, family = "CMU Sans Serif") +
  annotate("text", x = 6, y = 0.72, label = "BERT", size = 5.5, color = "#fe9929", fontface = "bold", family = "CMU Sans Serif") +
  annotate("text", x = 9, y = 0.72, label = "ELECTRA", size = 5.5, color = "#54278f", fontface = "bold", family = "CMU Sans Serif") +
  annotate("text", x = 16.5, y = 0.72, label = "GPT/GPT2", size = 5.5, color = "#93003a", fontface = "bold", family = "CMU Sans Serif") +
  annotate("text", x = 21.5, y = 0.72, label = "Eleuther AI\nGPT-Neo/J", size = 5.5, color = "#595959", fontface = "bold", family = "CMU Sans Serif") +
  annotate("text", x = 12, y = 0.72, label = "RoBERTa", size = 5.5, color = "#238443", fontface = "bold", family = "CMU Sans Serif") +
  annotate("text", x = 2, y = 0.88, label = "SotA", size = 5.5, color = "black", family = "CMU Sans Serif", fontface="bold") +
  annotate("text", x = 2, y = 0.97, label = "Human", size = 5.5, color = "black", family = "CMU Sans Serif", fontface="bold") +
  scale_color_identity(aesthetics = c("color", "fill")) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0.012), breaks = scales::pretty_breaks(6)) +
  theme_bw(base_size = 18, base_family = "CMU Serif") +
  theme(
    legend.position = "None",
    # axis.text.x = element_blank(),
    plot.margin = margin(0.2, 0.2, 0.1, 0.1, "cm"),
    axis.text.x = element_text(angle = 20, vjust = 0.5, color = "black", size = 15),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(family = "CMU Sans Serif", color = "black"),
    axis.title.x = element_blank(),
    panel.background = element_rect(fill = "transparent"), # bg of the panel
    plot.background = element_rect(fill = "transparent", color = NA), # bg of the plot
    legend.background = element_rect(fill = "transparent"), # get rid of legend bg
    legend.box.background = element_rect(fill = "transparent")
  ) +
  labs(y = "Accuracy")

p


ggsave("analysis/anlimodels.pdf", p,height = 4.5, width = 12, device = cairo_pdf, dpi = 300)
