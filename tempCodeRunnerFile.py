

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    # Example usage with a raw string
    X, y = load_data(r"D:\Notes\Phy\k\cursed\Cervical Cancer\data")
    print(f"Loaded {len(X)} images with labels: {set(y)}")