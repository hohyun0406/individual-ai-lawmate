package site.lawmate.chatbot.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VectorStoreService {
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    public void saveVector(String key, List<Double> vector) {
        redisTemplate.opsForValue().set(key, vector);
    }

    public List<String> similaritySearch(List<Double> queryVector, int topK) {
        return null; // Redis에서 유사한 벡터 찾는 로직 구현
    }

}
