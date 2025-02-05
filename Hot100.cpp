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

// 子集和等的问题使用背包问题的算法，是 NP-hard 问题。
class Solution {
public:
    bool canPartition_raw(vector<int>& nums) {
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

    // 空间优化
    bool canPartition(vector<int>& nums) {
        int total = accumulate(nums.begin(), nums.end(), 0);
        if(total % 2 != 0) return false;
        
        int target = total / 2;
        vector<bool> dp(target + 1, false);
        dp[0] = true;
        
        for(int num : nums) {
            for(int j = target; j >= num; j--) { // 从后向前数，可规避空间问题！
                dp[j] = dp[j] || dp[j - num];
            }
        }
        
        return dp[target];
    }
};
*/



/*
// T5 最长回文子串
// 给你一个字符串 s，找到 s 中最长的 回文子串。

// 朴素想法是遍历整个子串的头尾（O(N^2）)，但这样会有一些重复检测行为。我们可以通过动态规划记录下检测过的子串位置！
class Solution {
public:
    string longestPalindrome(string s) {
        int length = s.size();\
        vector<vector<int>> dp(length, vector<int>(length, 0)); // dp[i][j] 表示差为i,开头为j的位置的字符子串是否是回文子串
        for(int i = 0; i < length; i++) dp[0][i] = 1; // 初始化：头尾是同一个位置，一定是回文（一个字母）
        vector<int> res(2, 0); // 答案的差和开头
        for(int i = 0; i < length - 1; i++) 
        {   
            if(s[i] == s[i + 1])
            {
                dp[1][i] = 1;  // 初始化，两个相邻字母一样，一定是回文
                res[0] = 1;
                res[1] = i;
            }
        }

        
        for(int i = 2; i < length; i++) // 表示两者之差
        {
            for(int j = 0; j < length - i; j++) // 第一项
            {
                if(s[j] == s[j + i] && dp[i - 2][j + 1])
                {
                    dp[i][j] = 1; // 这一对也是
                    res[0] = i;
                    res[1] = j;
                }
            }
        }
        return s.substr(res[0], res[1] - res[0] + 1);
    }
};
*/


/*
// T1143 最长公共子序列
// 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
// 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
// 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
// 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

// 思路：dp[i][j] 表示 t1[0,...,i] 与 t2[0,...,j] 的最长公共子序列长度，然后进行动态规划。
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int len1 = text1.size(); int len2 = text2.size();
        vector<vector<int>> dp(len1, vector<int>(len2, 0)); // dp[i][j] 表示 t1[0,...,i] 与 t2[0,...,j] 的最长公共子序列长度
        bool flag = false; // 用于赋初值
        for(int j = 0; j < len2; j++)
        {
            if(text2[j] == text1[0]) flag = true;
            if(flag) dp[0][j] = 1; // 赋初值
        } 
        flag = false;
        for(int i = 0; i < len1; i++)
        {
            if(text1[i] == text2[0]) flag = true;
            if(flag) dp[i][0] = 1; // 赋初值
        } 

        for(int i = 1; i < len1; i++)
        {
            for(int j = 1; j < len2; j++)
            {
                dp[i][j] = max({dp[i - 1][j - 1] + (text1[i] == text2[j]), dp[i - 1][j], dp[i][j - 1]});
            }
        }

        return dp[len1 - 1][len2 - 1];
    }
};
*/



/*
// T72 编辑距离
// 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
// 你可以对一个单词进行如下三种操作：
// 插入一个字符
// 删除一个字符
// 替换一个字符

// 思路：dp[i][j] 表示 word1[0,,i] 和 word2[0,,j] 的编辑距离，进行动态规划。
class Solution {
public:
    int minDistance(string word1, string word2) {
        int len1 = word1.size(); int len2 = word2.size();
        vector<vector<int>> dp(len1 + 1, vector<int>(len2 + 1, 0)); // dp[i][j] 表示 word1[0,,i-1] 和 word2[0,,j-1] 的编辑距离
        for(int i = 0; i <= len1; i++) dp[i][0] = i; // 初始化第一列
        for(int j = 0; j <= len2; j++) dp[0][j] = j; // 初始化第一行

        for(int i = 1; i <= len1; i++)
        {
            for(int j = 1; j <= len2; j++)
            {
                if(word1[i - 1] == word2[j - 1]) dp[i][j] = dp[i - 1][j - 1]; // 如果结尾字母相等，则与 dp[i - 1][j - 1] 相等；
                else dp[i][j] = min({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]}) + 1; // 结尾字母不相等，则取这三个的最小值
            }
        }

        return dp[len1][len2];
    }
};
*/