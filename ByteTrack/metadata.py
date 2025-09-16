from pymediainfo import MediaInfo

media_info = MediaInfo.parse("./videos/Recv02_20250908045910@5802bc22.avi")

for track in media_info.tracks:
    print("====", track.track_type, "====")
    for key, value in track.to_data().items():
        print(f"{key}: {value}")
