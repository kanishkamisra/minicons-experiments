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

cor.test(test_res$parameters, test_res$accuracy)

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
    strip.text = element_text(family = "CMU Sans Serif", size = 11, face = "bold"),
    axis.title = element_text(family = "CMU Sans Serif"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in million)",
    y = "Accuracy"
  )

ggsave("analysis/anli.pdf", p, height = 2.7, width = 4.5, device = cairo_pdf, dpi = 300)


fs::dir_ls("data/results/anli/", regexp = "*_anli.csv") %>%
  map_df(read_csv) %>%
  distinct(model, parameters) %>% 
  write_csv("data/meta.csv")
