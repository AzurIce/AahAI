非极大值抑制



MAA 中 方向分类部分逻辑：[MaaAssistantArknights/src/MaaCore/Task/Experiment/CombatRecordRecognitionTask.cpp at dev · MaaAssistantArknights/MaaAssistantArknights (github.com)](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/dev/src/MaaCore/Task/Experiment/CombatRecordRecognitionTask.cpp#L655-L671)

```cpp
using Raw = BattlefieldClassifier::DeployDirectionResult::Raw;
    constexpr size_t ClsSize = BattlefieldClassifier::DeployDirectionResult::ClsSize;
    std::unordered_map<Point, Raw> dir_cls_sampling;

    for (const cv::Mat& frame : clip.random_frames) {
        BattlefieldClassifier analyzer(frame);
        analyzer.set_object_of_interest({ .skill_ready = false, .deploy_direction = true });
        for (const auto& loc : newcomer) {
            analyzer.set_base_point(m_normal_tile_info.at(loc).pos); // 后续会用到的 m_base_point
            auto result_opt = analyzer.analyze();
            show_img(analyzer);
            for (size_t i = 0; i < ClsSize; ++i) {
                dir_cls_sampling[loc][i] += result_opt->deploy_direction.raw[i];
            }
        }
    }
```

其中的 `m_normal_tile_nfo` 的计算在：[MaaAssistantArknights/src/MaaCore/Task/Experiment/CombatRecordRecognitionTask.cpp at dev · MaaAssistantArknights/MaaAssistantArknights (github.com)](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/dev/src/MaaCore/Task/Experiment/CombatRecordRecognitionTask.cpp#L221-L222)

```cpp
auto calc_result = Tile.calc(m_stage_name);
m_normal_tile_info = std::move(calc_result.normal_tile_info);
```

或者：[MaaAssistantArknights/src/MaaCore/Task/BattleHelper.cpp at 37aff294435a63d448095e20dedd3784a170b221 · MaaAssistantArknights/MaaAssistantArknights (github.com)](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/37aff294435a63d448095e20dedd3784a170b221/src/MaaCore/Task/BattleHelper.cpp#L57-L76)

具体的计算逻辑在：



