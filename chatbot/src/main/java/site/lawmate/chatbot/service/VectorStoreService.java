package site.lawmate.chatbot.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import site.lawmate.chatbot.VectorUtils;

import java.util.*;
import java.util.List;

@Service
public class VectorStoreService {
    private final RedisTemplate<String, Object> redisTemplate;

    @Autowired
    public VectorStoreService(RedisTemplate<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    // 벡터 저장 (개별 키로 저장)
    public void saveVector(String key, List<Double> vector) {
        redisTemplate.opsForValue().set(key, vector);
    }

    // 벡터 유사도 검색
    public List<String> similaritySearch(List<Double> queryVector, int topK) {
        // 1. 모든 벡터 데이터 불러오기 (실제 프로젝트에서는 최적화 필요)
        Set<String> keys = redisTemplate.keys("*");
        List<SearchResult> results = new ArrayList<>();

        if (keys != null) {
            for (String key : keys) {
                List<Double> documentVector = (List<Double>) redisTemplate.opsForValue().get(key);
                if (documentVector != null) {
                    double similarity = VectorUtils.cosineSimilarity(queryVector, documentVector);
                    results.add(new SearchResult(key, similarity));
                }
            }
        }

        // 2. 유사도 계산 및 상위 K개 결과 추출
        results.sort(Collections.reverseOrder(Comparator.comparing(SearchResult::getScore)));

        List<String> topKResults = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, results.size()); i++) {
            topKResults.add(results.get(i).getKey());
        }

        return topKResults;
    }

    private static class SearchResult {
        private final String key;
        private final double score;

        public SearchResult(String key, double score) {
            this.key = key;
            this.score = score;
        }

        public String getKey() {
            return key;
        }

        public double getScore() {
            return score;
        }
    }
}
