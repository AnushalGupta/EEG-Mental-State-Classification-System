-- EEG Features Table Schema
CREATE TABLE IF NOT EXISTS eeg_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(50),
    subject_id VARCHAR(100),
    label INT,  -- 0: Focused, 1: Unfocused, 2: Drowsy
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Feature Columns (Band Powers)
    ch1_delta FLOAT, ch1_theta FLOAT, ch1_alpha FLOAT, ch1_beta FLOAT,
    ch2_delta FLOAT, ch2_theta FLOAT, ch2_alpha FLOAT, ch2_beta FLOAT,
    ch3_delta FLOAT, ch3_theta FLOAT, ch3_alpha FLOAT, ch3_beta FLOAT,
    ch4_delta FLOAT, ch4_theta FLOAT, ch4_alpha FLOAT, ch4_beta FLOAT,
    ch5_delta FLOAT, ch5_theta FLOAT, ch5_alpha FLOAT, ch5_beta FLOAT,
    ch6_delta FLOAT, ch6_theta FLOAT, ch6_alpha FLOAT, ch6_beta FLOAT,
    ch7_delta FLOAT, ch7_theta FLOAT, ch7_alpha FLOAT, ch7_beta FLOAT,
    ch8_delta FLOAT, ch8_theta FLOAT, ch8_alpha FLOAT, ch8_beta FLOAT,
    ch9_delta FLOAT, ch9_theta FLOAT, ch9_alpha FLOAT, ch9_beta FLOAT,
    ch10_delta FLOAT, ch10_theta FLOAT, ch10_alpha FLOAT, ch10_beta FLOAT,
    ch11_delta FLOAT, ch11_theta FLOAT, ch11_alpha FLOAT, ch11_beta FLOAT,
    ch12_delta FLOAT, ch12_theta FLOAT, ch12_alpha FLOAT, ch12_beta FLOAT,
    ch13_delta FLOAT, ch13_theta FLOAT, ch13_alpha FLOAT, ch13_beta FLOAT,
    ch14_delta FLOAT, ch14_theta FLOAT, ch14_alpha FLOAT, ch14_beta FLOAT
);
