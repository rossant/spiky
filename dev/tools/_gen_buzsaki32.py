
    
    
geometrical_positions = np.array([
                                (0, 0),
                                (5, 10),
                                (-6, 20),
                                (7, 30),
                                (-8, 40),
                                (9, 50),
                                (-10, 60),
                                (11, 70),
                                (-12, 80),
                                (13, 90),
                                (-14, 100),
                                (15, 110),
                                (-16, 120),
                                (17, 130),
                                (-18, 140),
                                (19, 150),
                                (-20, 160),
                                (21, 170),
                                (-22, 180),
                                (23, 190),
                                (-24, 200),
                                (25, 210),
                                (-26, 220),
                                (27, 230),
                                (-28, 240),
                                (29, 250),
                                (-30, 260),
                                (31, 270),
                                (-32, 280),
                                (33, 290),
                                (-34, 300),
                                (35, 310)], dtype=np.float32)

np.savetxt("buzsaki32.txt", geometrical_positions, fmt='%d')
