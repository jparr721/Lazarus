library(readr)

setwd("~/ML/Lazarus/Diabetes-Data")

DIQ_I <- read_csv("csvs/DIQ_I.csv")
ALQ_I <- read_csv("csvs/ALQ_I.csv")
BPQ_I <- read_csv("csvs/BPQ_I.csv")
CDQ_I <- read_csv("csvs/CDQ_I.csv")
DIQ_I <- read_csv("csvs/DIQ_I.csv")
PAQ_I <- read_csv("csvs/PAQ_I.csv")
SMQ_I <- read_csv("csvs/SMQ_I.csv")
WHQ_I <- read_csv("csvs/WHQ_I.csv")

# Take all of the pain columns where missing means they didn't select that particular thing
CDQ_I_Pain <- CDQ_I[,c("CDQ009A", "CDQ009B", "CDQ009C", "CDQ009D", "CDQ009E", "CDQ009F", "CDQ009G", "CDQ009H")]

# Impute zeroes into the data so it can be read as selected or not
CDQ_I_Pain[is.nan(CDQ_I_Pain)]<- 0

# Merge the data
list_x15_16 <- list(DIQ_I, ALQ_I, BPQ_I, CDQ_I, DIQ_I, PAQ_I, SMQ_I, WHQ_I)
for ( frame in list_x15_16 ) {
  x15_16_ALL <- merge(x15_16_ALL, frame, by = "SEQN", all = TRUE
}
