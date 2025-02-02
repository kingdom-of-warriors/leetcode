#include <bits/stdc++.h>

using namespace std;

/*
// 动态规划
// 118. 杨辉三角
// 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res;
        res.push_back(vector<int>(1, 1)); // 添加第1项
        if(numRows == 1) return res;
        res.push_back(vector<int>(2, 1)); // 添加第2项
        if(numRows == 2) return res;

        for(int i = 2; i < numRows; i++)
        {
            vector<int> lastRow = res.back(); // 上一行的元素
            vector<int> row(i + 1, 1); // 赋初值，第一项和最后一项都是1
            for(int j = 1; j < i; j++)
            {
                row[j] = lastRow[j - 1] + lastRow[j];
            }
            res.push_back(row);
        }

        return res;
    }
};
*/



/*
// 139. 单词拆分
// 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。
// 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.length();
        // dp[i]表示s的前i个字符能否被拆分
        vector<bool> dp(n + 1, false);
        dp[0] = true; // 空字符串可以被拆分
        
        // 将字典转换为哈希集合，方便快速查找
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        
        for(int i = 1; i <= n; i++) {
            for(int j = 0; j < i; j++) {
                // 如果前j个字符可以被拆分，且s[j...i-1]在字典中
                if(dp[j] && wordSet.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break; // 找到一个可行的拆分方式即可
                }
            }
        }
        
        return dp[n];
    }
};
*/


/*
// T300 最长递增子序列
// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
// 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> res(nums.size(), 1); // res[i] 表示以 nums[i, ...] 这个数组，用到 nums[i] 的最长严格递增子序列的长度
        for(int i = nums.size() - 2; i >= 0; i--)
        {
            for(int j = i + 1; j <= nums.size() - 1; j++)
            {
                if (nums[j] > nums[i]) // 说明这一项可以作为后续
                {
                    res[i] = max(res[i], res[j] + 1);
                }
            }
        }
        return *max_element(res.begin(), res.end()); // 最大的那一项
    }
};
*/


/*
// T152 乘积最大子数组
// 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续 子数组，并返回该子数组所对应的乘积。
// 测试用例的答案是一个 32-位 整数。

class Solution {
public:
    int maxProduct_raw(vector<int>& nums) {
        vector<int> res_max = nums; // res[i] 表示第一项为 nums[i] 的乘积最大的非空连续子数组的乘积
        vector<int> res_min = nums; // res[i] 表示第一项为 nums[i] 的乘积最小的非空连续子数组的乘积
        for(int i = nums.size() - 2; i >= 0; i--)
        {
            res_max[i] = max({nums[i], res_max[i + 1] * nums[i], res_min[i + 1] * nums[i]});
            res_min[i] = min({nums[i], res_min[i + 1] * nums[i], res_max[i + 1] * nums[i]});
        }
        return *max_element(res_max.begin(), res_max.end()); // 最大的那一项
    }

    // 滚动数组优化
    int maxProduct(vector<int>& nums) {
        int max_now = nums.back(), max_pre = nums.back(), min_now = nums.back(), min_pre = nums.back();
        int res = nums.back(); // 答案
        for(int i = nums.size() - 2; i >= 0; i--)
        {
            max_pre = max_now; min_pre = min_now; // 存储前一个最大值和最小值
            max_now = max({nums[i], max_pre * nums[i], min_pre * nums[i]});
            min_now = min({nums[i], max_pre * nums[i], min_pre * nums[i]});
            res = max(res, max_now);
        }
        return res; // 最大的那一项
    }
};
*/



/*
// T416 分割等和子集
// 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int total = accumulate(nums.begin(), nums.end(), 0);
        if (total % 2) return false; // 和不为 2 的倍数
        int target = total / 2; // 目标

        // 背包问题
        vector<vector<int>> dp(nums.size(), vector<int>(target + 1)); // 初始化，dp[i][j] 表示前i个数字是否存在某些和为j
        dp[0][0] = true; 
        if(nums[0] <= target) dp[0][nums[0]] = true; // 初始化
        for(int i = 1; i < nums.size(); i++)
        {
            for(int j = 0; j <= target; j++)
            {
                if (j >= nums[i]) dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]]; // 不选i和选i
                else dp[i][j] = dp[i - 1][j];
            }
        }

        return dp[nums.size() - 1][target];
    }
};
*/