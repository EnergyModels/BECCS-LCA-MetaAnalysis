library(dplyr)
library("readxl")

#===================================================================
# Case-based results
#===================================================================

# Set directory to load data
dir = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir)

# Load data
data <- read_excel("model.xlsx", sheet = "Summary")

# transpose data
data <- t(data)

# store in data.frame
df = data.frame(data)

# set heading as first row
names(df) <- data[c(1),]

# remove rows
df <- df[-c(1, 2, 3), ]

# remove entries without capture
df<-df[!(df$captureType1=="None"),]
     
# set entires as numeric data
df$GWP_total <- as.numeric(as.character(df$GWP_total))
df$EROI <- as.numeric(as.character(df$EROI))
df$WU_total <- as.numeric(as.character(df$WU_total))
df$EU_total <- as.numeric(as.character(df$EU_total))


# Summarise data
df_smry <- df %>% # the names of the new data frame and the data frame to be summarised
  group_by(.dots=c("powerPlantType","captureType1", "PrePostOxy")) %>%   # the grouping variable
  summarise(count=n(), # counts number of entries
            GWP_min = min(GWP_total), # calculates the minimum
            GWP_mean = mean(GWP_total),  # calculates the mean
            GWP_max = max(GWP_total),# calculates the maximum
            EROI_min = min(EROI),
            EROI_mean = mean(EROI),
            EROI_max = max(EROI),
            WU_min = min(WU_total),
            WU_mean = mean(WU_total),
            WU_max = max(WU_total),
            EU_min = min(EU_total),
            EU_mean = mean(EU_total),
            EU_max = max(EU_total),
            )

# Convert from g to kg
df_smry$GWP_min <- df_smry$GWP_min / 1000.0
df_smry$GWP_mean <- df_smry$GWP_mean / 1000.0
df_smry$GWP_max <- df_smry$GWP_max / 1000.0

# Convert from cm^3 to l
df_smry$WU_min <- df_smry$WU_min / 1000.0
df_smry$WU_mean <- df_smry$WU_mean / 1000.0
df_smry$WU_max <- df_smry$WU_max / 1000.0

# save data
write.csv(df_smry, "summary.csv")
