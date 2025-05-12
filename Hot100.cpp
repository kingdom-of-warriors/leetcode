#include <bits/stdc++.h>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

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


// 二叉树


/*
// 94. 二叉树的中序遍历
// 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

class Solution_recursion {
public:
    vector<int> res;
    void inorder(TreeNode* root) {
        if(!root) return;  // 修正：inorder函数是void类型，不应该返回res

        inorder(root->left);
        res.push_back(root->val);
        inorder(root->right);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        inorder(root);
        return res;
    }
};
*/


/*
// T104. 二叉树的最大深度
// 给定一个二叉树 root ，返回其最大深度。
// 二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
*/


// T226 翻转二叉树
// class Solution {
// public:
//     void invert(TreeNode* root){
//         if(!root) return;

//         invert(root->left);
//         invert(root->right);
//         swap(root->left, root->right);
//     }

//     TreeNode* invertTree(TreeNode* root) {
//         invert(root);
//         return root;
//     }
// };



/*
// T101 对称二叉树
class Solution {
public:
    bool isSymmetric_2(TreeNode* root1, TreeNode* root2) { // 判断以root1, root2为根节点的子树是否对称
        if(!root1 && !root2) return true; // 两个空节点
        if((!root1 && root2) || (root1 && !root2)) return false; // 一空一非空
        return (root1->val == root2->val) && isSymmetric_2(root1->left, root2->right) && isSymmetric_2(root1->right, root2->left);
        // 根节点相等，左节点左树和右节点右树对称，左节点右树和右节点左树对称
    }

    bool isSymmetric(TreeNode* root) {
        return isSymmetric_2(root->left, root->right);
    }
};
*/


/*
// T543 二叉树的直径
// 给你一棵二叉树的根节点，返回该树的 直径 。
// 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
// 两节点之间路径的 长度 由它们之间边数表示。

// 经过某个点的直径就是左子树深度加右子树深度！
class Solution {
public:
    int res = 0;
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        int L = maxDepth(root->left);
        int R = maxDepth(root->right);
        res = max(res, L + R); // 计算深度同时更新 res
        return max(L, R) + 1;
    }

    int diameterOfBinaryTree(TreeNode* root) {
        int tmp = maxDepth(root);
        return res;
    }
};
*/



/*
// T230 二叉搜索树的第k小元素
// 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。

// 思路：用栈实现中序遍历，取第k项，用栈详见 https://blog.csdn.net/qq_43753525/article/details/102905590
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> s; // 栈，用于中序遍历
        while(root || !s.empty())
        {
            if(root)
            {
                s.push(root);
                root = root->left;
            }class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        
    }
};
            else
            {
                root = s.top();
                k--;
                if(k == 0) return root->val;
                s.pop();
                root = root->right;
            }
        }
        return -1;
    }
};
*/


/*
// T199 二叉树的右视图
// 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
// 思路：返回每一行的最右边的元素，参考二叉树的层序遍历。
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        TreeNode* eof = new TreeNode(-1); // 特殊节点，表示这一行结束
        vector<int> res; // 答案数组
        if(!root) return res; // 特殊情况

        queue<TreeNode*> q;
        q.push(root); // 根部进队
        q.push(eof); // 第一行结束

        TreeNode* a; // 用于遍历
        TreeNode* pre; // 保存遍历的上一个节点
        while(!q.empty())
        {
            a = q.front(); // 取出头部
            q.pop();
            if(a == eof) // 如果是特殊字符，那么说明这一行结束
            {
                res.push_back(pre->val);
                if(q.empty()) break; // 若除掉特殊字符没有别的元素，说明遍历完成
                q.push(eof); // 往队列里加入eof，这说明这一行元素没有了
            }
            pre = a;
            if(a->left) q.push(a->left);
            if(a->right) q.push(a->right);
        }
        return res;
    }
};
*/


/*
// T114 二叉树展开为列表
// 给你二叉树的根结点 root ，请你将它展开为一个单链表：
// 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
// 展开后的单链表应该与二叉树 先序遍历 顺序相同。

// 思路：空间 O(N)，用栈实现前序遍历，未成功，等学完链表再来写这道题！
class Solution {
public:
    TreeNode* flatten(TreeNode* root) { 
        TreeNode* res;
        TreeNode* head = new TreeNode(-1, nullptr, res); // 头节点
        if(!root) return root;

        stack<TreeNode*> s;
        TreeNode* p = root; // 用于遍历
        while(!s.empty() || p)
        {
            if(p) // p 不为空
            {
                s.push(p);
                res = new TreeNode(p->val);
                res = res->right;
                p = p->left;
            }
            else // p为空指针
            {
                p = s.top();
                s.pop();
                p = p->right;
            }

        }
        root = head->right;
        return root;
    }
};

int main() {
    // 创建一个示例二叉树
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(5);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(4);
    root->right->right = new TreeNode(6);

    Solution solution;
    TreeNode* flattened = solution.flatten(root);

    // 打印展开后的链表
    while(flattened) {
        cout << flattened->val << " ";
        flattened = flattened->right;
    }
    cout << endl;

    return 0;
}
*/


/*
// T437 路径总和
// 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
// 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

// 思路：通过树结构进行递归。
class Solution {
public:
    long long res = 0;
    void pathSum_1(TreeNode* root, long long targetSum, bool isFetch) {
        if(root->val == targetSum) res++;
        if(isFetch) // 已经选取过
        {
            
            long long tmp = targetSum - root->val; // 防止溢出
            if(root->right) pathSum_1(root->right, tmp, true); // 若已经选过，那么必须选，否则就不是连续的
            if(root->left) pathSum_1(root->left, tmp, true); // 若已经选过，那么必须选，否则就不是连续的
        }
        else  // 未选取过
        {
            if(root->right)
            {
                pathSum_1(root->right, targetSum, false); // 不选
                pathSum_1(root->right, targetSum - root->val, true); // 选
            }
            if(root->left)
            {
                pathSum_1(root->left, targetSum, false);
                pathSum_1(root->left, targetSum - root->val, true);
            }
        }
    }

    long long pathSum(TreeNode* root, int targetSum) {
        if(!root) return 0;
        pathSum_1(root, targetSum, false);
        return res;
    }
};
*/



/*
// T236 二叉树的最近公共祖先
// 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
// 百度百科中最近公共祖先的定义为："对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。"
class Solution {
public:
    bool isInTree(TreeNode* root, TreeNode* p) // 判断p是否在以root为根的子树上
    {
        if(!root) return false;
        if(root == p) return true;
        return isInTree(root->left, p) || isInTree(root->right, p); // 要么在左子树，要么在右子树
    }

    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(p == root || q == root) return root;

        if(isInTree(root->left, p) && isInTree(root->left, q)) // 都在左子树上
        {
            return lowestCommonAncestor(root->left, p, q);
        }
        else if(isInTree(root->right, p) && isInTree(root->right, q)) // 都在右子树上
        {
            return lowestCommonAncestor(root->right, p, q);
        }
        else return root; // 一个在左，一个在右
    }

    TreeNode* lowestCommonAncestor_2(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return nullptr;
        if(p == root || q == root) return root;

        TreeNode* res_left = lowestCommonAncestor_2(root->left, p, q); // 在不在左子树上
        TreeNode* res_right = lowestCommonAncestor_2(root->right, p, q); // 在不在右子树上
        if(res_left && res_right) return root; // 说明两个点一个在左子树上一个在右子树上
        if(res_left == nullptr) return res_right;
        if(res_right == nullptr) return res_left;
        return NULL;
    }
};
*/




// 链表

/*
// T160 相交链表
// 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

// 思路：全部入栈，从后向前出栈
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        stack<ListNode *> s1, s2;
        ListNode *p, *q, *res; // 两个用于遍历的指针，一个答案指针
        p = headA; q = headB; // 赋初值
        while(p) {s1.push(p);p = p->next;}
        while(q) {s2.push(q);q = q->next;}

        if(s1.top() != s2.top()) return NULL;
        while(s1.top() == s2.top())
        {
            res = s1.top();
            s1.pop(); s2.pop(); // 不是这一项
            if(s1.empty() || s2.empty()) return res; // 但凡有一个空了，都要返回
        }
        return res;
    }
};
*/


/*
// T206 反转链表
// 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

// 思路：栈
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head) return nullptr; // 空链表特判
        
        stack<ListNode*> s;
        ListNode *p = head; // 用于遍历
        while(p) {s.push(p); p = p->next;}

        ListNode *new_head = s.top(); // 新头
        s.pop();
        p = new_head; // 这里重复定义了p变量,应该直接使用上面的p
        while(!s.empty())
        {
            p->next = s.top();
            s.pop();
            p = p->next;
        }
        p->next = nullptr; // 最后一个节点的next需要置空,否则可能形成环
        return new_head;
    }
};
*/


/*
// T234 回文链表
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if(!head) return true;
        vector<int> l;
        ListNode *p = head;
        while(p)
        {
            l.push_back(p->val);
            p = p->next;
        }
        if(l.size() == 1) return true; // 只有一项
        for(int i = 0; i <= l.size() / 2; i++)
        {
            if(l[i] != l[l.size() - i - 1]) return false;// 有一个不相等，返回false
        }
        return true;
    }
};
*/


/*
// T141 环形链表1\2
// 给你一个链表的头节点 head ，判断链表中是否有环。
// 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

// 环形链表1：如果链表中存在环 ，则返回 true 。 否则，返回 false 。
// 环形链表2：返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
// 思路：快慢指针，方法2：哈希表记录第一个重复的节点

class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head) return false;
        ListNode *fast = head, *slow = head; // 快慢指针
        while(true)
        {
            if(!fast->next || !slow->next) return false; // 只要不能继续，一定是没有环的
            if(!fast->next->next) return false; // 只要不能继续，一定是没有环的
            fast = fast->next->next;
            slow = slow->next;
            if(slow == fast) return true;
        }
        return false;
    }

    ListNode *detectCycle(ListNode *head) {
        if(!head) return nullptr;
        unordered_map<ListNode*, int> hash_map; // 用来记录第一个重复的点
        ListNode *p = head;  // 用于遍历
        while(p) // 修改：使用p作为循环条件而不是true
        {
            if(hash_map[p] == 1) // 修改：检查是否已经访问过
            {
                return p;
            }
            hash_map[p]++;
            p = p->next;
        }
        return nullptr;
    }
};
*/


/*
// T21 合并两个有序链表
// 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *p = list1, *q = list2; // 两个用于遍历的指针
        ListNode *new_head = new ListNode(0);
        ListNode *k = new_head; // 也是用于遍历的指针

        while(p && q) // 当两个都不为空指针时
        {
            if(p->val <= q->val)
            {
                ListNode *new_node = new ListNode(p->val);
                k->next = new_node;
                k = k->next;
                p = p->next;
            }
            else
            {
                ListNode *new_node = new ListNode(q->val);
                k->next = new_node;
                k = k->next;
                q = q->next;
            }
        }
        // 以下两个 while 最多只有一个被运行
        while(p)
        {
            ListNode *new_node = new ListNode(p->val);
            k->next = new_node;
            k = k->next;
            p = p->next;
        }
        while(q)
        {
            ListNode *new_node = new ListNode(q->val);
            k->next = new_node;
            k = k->next;
            q = q->next;
        }

        return new_head->next;
    }
};
*/


/*
// T2 两数相加
// 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
// 请你将两个数相加，并以相同形式返回一个表示和的链表。
// 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *p = l1, *q = l2; // 用于遍历两个链表
        ListNode *new_head = new ListNode(0); // 新链表，用于储存答案
        ListNode *k = new_head; // 用于延伸答案链表
        
        bool is_carry = false; // 是否进位
        while(p && q)
        {
            int sum = p->val + q->val + is_carry;
            if(sum >= 10) 
            {
                sum -= 10;
                is_carry = true;
            }
            else is_carry = false;
            ListNode *new_node = new ListNode(sum); // 建立新节点
            k->next = new_node;
            k = k->next; p = p->next; q = q->next; // 更新三个遍历指针
        }
        // 以下两个 while 最多只有一个被运行
        while(p)
        {
            int sum = p->val + is_carry;
            if(sum >= 10) 
            {
                sum -= 10;
                is_carry = true;
            }
            else is_carry = false;
            ListNode *new_node = new ListNode(sum); // 建立新节点
            k->next = new_node;
            k = k->next; p = p->next;
        }
        while(q)
        {
            int sum = q->val + is_carry;
            if(sum >= 10) 
            {
                sum -= 10;
                is_carry = true;
            }
            else is_carry = false;
            ListNode *new_node = new ListNode(sum); // 建立新节点
            k->next = new_node;
            k = k->next; q = q->next;
        }

        if(is_carry) // 最后一位也要进位，前面要多一个1: 如 999 + 1 = 1000
        {
            ListNode *new_node = new ListNode(1); // 建立新节点
            k->next = new_node;
        }

        return new_head->next;
    }
};
*/


/*
// T19 删除链表的倒数第N个节点
// 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        vector<ListNode*> l; // 用于储存链表元素
        ListNode *p = head; // 用于遍历
        while(p)
        {
            l.push_back(p);
            p = p->next;
        }
        // 分三种情况，头、尾、中间
        if(l.size() == 1) return nullptr; // 只有一个元素，那么它一定被删掉
        if(n == 1) // 要删除的是尾
        {
            l[l.size() - 2]->next = nullptr; // 断联
            return head;
        }
        if(n == l.size()) return head->next; // 要删除的是头
        // 要删除的是中间
        l[l.size() - n - 1]->next = l[l.size() - n + 1];
        return head;
    }
};
*/

/*
// T19 两两交换链表中的节点
// 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

// 思路：显然是利用递归，一次交换一对相邻的节点
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(!head || !head->next) return head;  // 如果链表为空或只有一个节点，直接返回
        // 交换前两个
        ListNode *tmp = head->next;  // 保存第二个节点
        head->next = tmp->next;      // 第一个节点指向第三个节点
        tmp->next = head;            // 第二个节点指向第一个节点
        
        tmp->next->next = swapPairs(tmp->next->next);  // 递归处理剩余的节点对
        return tmp;  // 返回新的头节点（原来的第二个节点）
    }
};
*/


/*
// 25. K 个一组翻转链表
// 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
// k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
// 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if(k == 1) return head;                          // 如果k=1则无需翻转，直接返回头节点
        int i = k;                                       // 计数器，用于检查是否有k个节点
        ListNode *p = head;                              // 遍历指针
        vector<ListNode*> l;                             // 存储k个节点的数组
        while(i--)                                       // 尝试获取k个节点
        {
            if(!p) return head;                          // 如果不足k个节点，保持原序返回
            l.push_back(p);                              // 将节点存入数组
            p = p->next;                                 // 移动到下一个节点
        }
        ListNode *tmp = l[k - 1]->next;                  // 保存下一组的起始节点
        for(int i = k - 1; i >= 1; i--) l[i]->next = l[i - 1];  // 反转当前k个节点
        l[0]->next = reverseKGroup(tmp, k);              // 递归处理剩余节点
        return l[k - 1];                                 // 返回当前组反转后的头节点
    }
};
*/


/*
// T138 随机链表的复制
// 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
// 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
// 例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
// 返回复制链表的头节点。
// 用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
// val：一个表示 Node.val 的整数。
// random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
// 你的代码 只 接受原链表的头节点 head 作为传入参数。

// 思路：创建节点和位置索引的双向哈希表，
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val, Node* _next, Node* _random) {
        val = _val;
        next = _next;
        random = _random;
    }
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(!head) return head;  // 如果链表为空，直接返回空指针
        Node *p = head; // 用于遍历老链表的指针p指向头节点
        Node *k = new Node(p->val, nullptr, nullptr); // 创建新链表的头节点，值与原链表头节点相同
        Node *q = k; // 保存新链表的头节点位置，用于最后返回
        unordered_map<Node*, Node*> hash; // 创建哈希表，用于建立原节点到新节点的映射关系
        while(p)  // 第一次遍历：复制所有节点并建立next连接
        {
            Node* new_node = new Node(p->val, nullptr, nullptr);  // 创建新节点，值与原节点相同
            k->next = new_node;  // 将新节点连接到新链表中
            hash[p] = new_node;  // 在哈希表中记录原节点到新节点的映射
            k = k->next;  // 移动新链表的指针
            p = p->next;  // 移动原链表的指针
        }
        p = head;  // 重置指针p到原链表头部
        while(p)  // 第二次遍历：建立random指针的连接
        {
            hash[p]->random = hash[p->random];  // 利用哈希表，将新节点的random指向对应的新节点
            p = p->next;  // 移动到下一个节点
        }
        return q->next;  // 返回新链表的真正头节点（跳过第一个哨兵节点）
    }
};
*/



/*
// T148 排序链表
// 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
// 思路：当作数组一样归并排序
class Solution {
public:
    // vector<ListNode*> l; // 用于储存节点，相当于转化为数组的排序
    // ListNode* sortList(ListNode* head) {
    //     if(!head) return head; // 如果头节点为空，直接返回
    //     ListNode *p = head; // 定义指针p指向头节点
    //     while(p) // 遍历链表
    //     {
    //         l.push_back(p); // 将节点添加到vector中
    //         p = p->next; // 指针后移
    //     }
    //     merge_sort(0, l.size() - 1); // 归并排序
    //     return head; // 返回头节点

    // }
    // void merge_sort(int begin, int end) // 归并排序，用于递归
    // {
    //     if(begin == end) return; // 递归终止条件
    //     int mid = (end - begin) / 2 + begin; // 取出中间节点，begin-mid为一组，mid+1-end为一组，默认这两组已经有序
    //     merge_sort(begin, mid); // 递归排序左半部分
    //     merge_sort(mid + 1, end); // 递归排序右半部分
    //     int i = begin, j = mid + 1, k = 0; // 定义三个指针，i指向左半部分起始，j指向右半部分起始，k指向临时数组起始
    //     vector<int> tmp(end - begin + 1); // 定义临时数组
    //     while(i <= mid && j <= end) // 当左右两边都有元素时
    //     {
    //         if(l[i]->val <= l[j]->val) tmp[k++] = l[i++]->val; // 如果左边小于等于右边，将左边放入临时数组
                
    //         else tmp[k++] = l[j++]->val; // 否则将右边放入临时数组
    //     }
    //     while(i <= mid) tmp[k++] = l[i++]->val; // 将左边剩余元素放入临时数组
    //     while(j <= end) tmp[k++] = l[j++]->val; // 将右边剩余元素放入临时数组
    //     for(int i = begin; i <= end; i++) l[i]->val = tmp[i - begin]; // 将临时数组赋值回原数组
    // }

    // 再写一个链表版
    ListNode* sortList(ListNode* head) {
        if(!head) return head; // 如果头节点为空，直接返回
        ListNode *tail = head;
        while(tail->next) tail = tail->next;
        merge_sort_link_list(head, tail);
        return head;
    }

    void merge_sort_link_list(ListNode *begin, ListNode *end)
    {
        if(begin == end) return;
        ListNode *slow = begin, *fast = begin; // 快慢指针找中间节点
        while(fast != end)
        {
            fast = fast->next;
            if(fast == end) break;
            else 
            {
                fast = fast->next;
                slow = slow->next;
            }
        }
        merge_sort_link_list(begin, slow); // 当作这两组是有序的
        merge_sort_link_list(slow->next, end);
        // 开始合并
        ListNode *p = begin, *q = slow->next;
        vector<int> tmp; // 用于记录接下来的有序链表的值
        while(p != slow->next && q != end->next)
        {
            if(p->val <= q->val)
            {
                tmp.push_back(p->val);
                p = p->next;
            }
            else
            {
                tmp.push_back(q->val);
                q = q->next;
            }
        }
        while(p != slow->next)
        {
            tmp.push_back(p->val);
            p = p->next;
        }
        while(q != end->next)
        {
            tmp.push_back(q->val);
            q = q->next;
        }
        p = begin;
        for(int i = 0; i < tmp.size(); i++)
        {
            p->val = tmp[i];
            p = p->next;
        }
    }
};
*/


/*
// T23 合并 k 个升序链表

// 给你一个链表数组，每个链表都已经按升序排列。
// 请你将所有链表合并到一个升序链表中，返回合并后的链表。

// 思路：和合并两个有序链表一样来写
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int k = lists.size(); // 要合并链表的个数
        vector<ListNode*> p(k, nullptr); // 遍历时用的节点
        ListNode *head = new ListNode(0); // 哨兵节点
        ListNode *q = head; // 用于接链表
        for(int i = 0; i < k; i++) p[i] = lists[i]; // 赋初值
        
        while(true)
        {
            bool flag = true;
            for(int i = 0; i < k; i++)
            {
                if(!p[i]) 
                {
                    flag = false;
                    break;
                }
            }
            if(!flag) break;
            else
            {
                auto min_iter = min_element(p.begin(), p.end(), [](ListNode* a, ListNode* b) {return a->val <= b->val;});
                ListNode *new_node = new ListNode((*min_iter)->val);
                q->next = new_node; q = q->next;
                
                int min_idx = distance(p.begin(), min_iter);
                p[min_idx] = p[min_idx]->next;
            }
        }
        
    }
};
*/


/*
// T283 移动零
// 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
// 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int idx = 0; // 表示下一个非零元素应该放置的位置
        for(int i = 0; i < nums.size(); i++) // 遍历整个数组
        {
            if(nums[i] != 0) nums[idx++] = nums[i]; // 如果当前元素不是0，则将其移动到idx位置，并将idx加1
        }
        for(; idx < nums.size(); idx++) nums[idx] = 0; // 将idx之后的所有元素设置为0
    }
};
*/


// 栈


/*
// T20 有效的括号
// 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
// 有效字符串需满足：
// 左括号必须用相同类型的右括号闭合。
// 左括号必须以正确的顺序闭合。
// 每个右括号都有一个对应的相同类型的左括号。

class Solution {
public:
    bool isValid(string s) {
        stack<char> stk; int i = 0;
        while(i != s.size())
        {
            if(stk.empty()) 
            {
                stk.push(s[i++]);
                continue;
            }
            if(s[i] == ')')
            {
                char t = stk.top();
                if(t == '(') stk.pop();
                else stk.push(s[i]);
            }
            else if(s[i] == ']')
            {
                char t = stk.top();
                if(t == '[') stk.pop();
                else stk.push(s[i]);
            }
            else if(s[i] == '}')
            {
                char t = stk.top();
                if(t == '{') stk.pop();
                else stk.push(s[i]);
            }
            else stk.push(s[i]);
            i++;
        }
        return stk.empty();
    }
};
*/


// T128 最长连续序列
// 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
// 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

// class Solution {
// public:
//     int longestConsecutive(vector<int>& nums) {
//         if(nums.empty()) return 0;
//         unordered_map<int, int> hash_map;
//         for(int i = 0; i < nums.size(); i++) hash_map[nums[i]]++;
//         // int min_value = *min_element(nums.begin(), nums.end()); // 取得最小值
//         int res = 1; int max_res = -1;

//         for(auto it = hash_map.begin(); it != hash_map.end(); ++it)
//         {
//             if(it->second == -1) continue;
//             else
//             {
//                 int tmp = it->first + 1;
//                 while(hash_map.count(tmp))
//                 {
//                     res++;
//                     tmp++;
//                     hash_map[tmp] = -1;
//                 }
//                 max_res = max(res, max_res);
//                 res = 1;
//             }
//         }
//         return max_res;
//     }
// };

// int main()
// {
//     cout << "请输入n: ";
//     int n; cin >> n;
//     vector<int> F(n + 1); // F(n) 为得到n的最小操作数
//     for(int i = 1; i <= n; i++) F[i] = ((i % 2) ? F[i - 1] : min(F[i / 2], F[i - 1])) + 1;
//     cout << "至少需要的操作数为："<< F[n] << endl;
//     return 0;
// }



// 二分查找


/*
// T35 搜索插入位置
// 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
// 请必须使用时间复杂度为 O(log n) 的算法。
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        return searchInsertInVector(nums, target, 0, nums.size() - 1);
    }

    // 表示在[begin, end]之间应该插入哪里，满足numsbegin<= target <= numsend
    int searchInsertInVector(vector<int>& nums, int target, int begin, int end) 
    {
        if(begin == end)
        {
            if(target <= nums[begin]) return begin;
            else return begin + 1;
        }
        if(begin + 1 == end)
        {
            if(target <= nums[begin]) return begin;
            if(target > nums[end]) return end + 1;
            else return end;
        }
        int mid = (end - begin) / 2 + begin;
        if(nums[mid] == target) return mid;
        if(nums[mid] < target) return searchInsertInVector(nums, target, mid, end);
        if(nums[mid] > target) return searchInsertInVector(nums, target, begin, mid);
        return 0;
    }

    // 下面是标准的二分查找代码
    int searchInsert_1(vector<int>& nums, int target)
    {
        int left = 0, right = nums.size() - 1;
        while(left <= right)
        {
            int mid = (left - right) / 2 + right;
            if(target == nums[mid]) return mid;
            else if(target > nums[mid]) left = mid + 1;
            else right = mid - 1;
        }
        return left;
    }
};
*/



/*
// T74 搜索二维矩阵
// 给你一个满足下述两条属性的 m x n 整数矩阵：
// 每行中的整数从左到右按非严格递增顺序排列。
// 每行的第一个整数大于前一行的最后一个整数。
// 给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。

// 思路：两种方法：当作一维数组来找复杂度为log(MN)，二是先找行再找列，复杂度也为log(M)+log(N)=log(MN)
class Solution {
public:
    int row; int col;
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        row = matrix.size(); col = matrix[0].size();
        int left = 0, right = row * col - 1;
        while(left <= right)
        {
            int mid = (right - left) / 2 + left;
            if(target == matrix[mid / col][mid % col]) return true;
            else if(target < matrix[mid / col][mid % col]) right = mid - 1;
            else left = mid + 1;
        }
        return false;
    }
};
*/



/*
// T34. 在排序数组中查找元素的第一个和最后一个位置
// 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
// 如果数组中不存在目标值 target，返回 [-1, -1]。
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> res(2, -1);
        if(nums.empty()) return res;
        int left1 = 0, right1 = nums.size() - 1;
        // 先找右边界
        while(left1 <= right1)
        {
            int mid = (right1 - left1) / 2 + left1;
            if(target >= nums[mid]) left1 = mid + 1;
            else right1 = mid - 1;
        }
        // 此时 right1 为右边界

        int left2 = 0, right2 = nums.size() - 1;
        // 再找左边界
        while(left2 <= right2)
        {
            int mid = (right2 - left2) / 2 + left2;
            if(target <= nums[mid]) right2 = mid - 1;
            else left2 = mid + 1;
        }
        // 此时 left2 为左边界
        if(right1 >= 0 && right1 < nums.size() && left2 >= 0 && left2 < nums.size()) // 排除索引在数组之外
        {
            if(nums[right1] == target) return vector<int>{left2, right1}; // 确定数组中有 target 元素
            return res;
        }
        return res;
    }
};
*/


/*
// T33 搜索旋转排序数组
// 整数数组 nums 按升序排列，数组中的值 互不相同 。
// 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
// 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
// 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

// 思路：先写一个log复杂度的函数来判断一个序列是否有序
class Solution {
public:
    // 判断nums[left, right]是否是递增的
    bool isOrdered(vector<int>& nums, int left, int right)
    {
        if(left == right) return true;
        if(right - left == 1) return (nums[left] <= nums[right]);

        int mid = (right - left) / 2 + left;
        if(nums[mid] >= nums[left] && nums[mid] <= nums[right]) return isOrdered(nums, left, mid) && isOrdered(nums, mid, right);
        else return false;
    }

    // 下面是标准的二分查找代码
    int searchInsert(vector<int>& nums, int target, int left, int right)
    {
        int a = left, b = right, res = 0;
        while(left <= right)
        {
            int mid = (left - right) / 2 + right;
            if(target == nums[mid]) return mid;
            else if(target > nums[mid]) left = mid + 1;
            else right = mid - 1;
        }
        res = left;
        
        // nums[res]!=target和res<a和res>b都说明没找到，返回-1
        if(res < a || res > b) return -1;
        if(nums[res] != target) return -1;
        else return res;
    }

    int search_1(vector<int>& nums, int target, int left, int right) {
        if(left == right) return (nums[left] == target) ? left : -1;
        if(right - left == 1)
        {
            if(nums[left] == target) return left;
            if(nums[right] == target) return right;
            else return -1;
        }

        int mid = (right - left) / 2 + left;
        int idx;
        // [left, mid]和[mid, right] 肯定有一个是有序的，也有可能两个都有序
        bool s1 = isOrdered(nums, left, mid);
        bool s2 = isOrdered(nums, mid, right);

        if(s1) 
        {
            idx = searchInsert(nums, target, left, mid);
            if(idx != -1) return idx; // 说明找到了
            else // 说明不在这一半
            {
                return search_1(nums, target, mid, right);
            }
        }
        else
        {
            idx = searchInsert(nums, target, mid, right);
            if(idx != -1) return idx; // 说明找到了
            else // 说明不在这一半
            {
                return search_1(nums, target, 0, mid);
            }
        }
    }

    int search(vector<int>& nums, int target) {
        return search_1(nums, target, 0, nums.size() - 1);
    }
};
*/


/*
// T4 寻找两个正序数组的中位数
// 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
// 算法的时间复杂度应该为 O(log (m+n)) 。

// 思路：找到分割线 
// 1 2 5     1 2 |  5    ||    1 3 5 7     1 3  |  5 7
// 3 4       3   |  4    ||    2 4           2  |  4
// 1  2 |3  5
// 4  6 |7  8
class Solution {
public:
    int len1, len2, min_len, res;
    int is_valid(vector<int>& nums1, vector<int>& nums2, int mid) {
        int mid_2 = res - mid; // 长序列左边有几个数
        // 处理特殊情况
        if(mid == len1)
        {
            if(nums1[len1 - 1] <= nums2[mid_2]) return 0;
            else return -1; // 表示mid太大了
        }
        if(mid == 0)
        {
            if(nums1[0] >= nums2[mid_2 - 1]) return 0;
            else return 1; // 表示mid太小了
        }
        if(nums1[mid - 1] <= nums2[mid_2] && nums2[mid_2 - 1] <= nums1[mid]) return 0;
        if(nums1[mid - 1] > nums2[mid_2]) return -1; // 表示mid太大
        else return 1; // 表示mid太小
    }

    double findDivider(vector<int>& nums1, vector<int>& nums2) {
        // 处理空数组情况
        if(nums1.empty()) return (nums2.size() % 2) ? nums2[nums2.size() / 2] : double(nums2[nums2.size() / 2] + nums2[nums2.size() / 2 - 1]) / 2;

        len1 = nums1.size();
        len2 = nums2.size();
        res = (len1 + len2 + 1) / 2;
        // 默认nums1.size() <= nums2.size()
        // 可选择的机会为 0,1,...,min_len, 代表短的序列左边有几个数
        int left = 0, right = len1; // left 代表左边有几个数
        // 标准二分代码
        while(left <= right)
        {
            int mid = (right - left) / 2 + left;
            if(is_valid(nums1, nums2, mid) == 0)
            {
                left = mid;
                break;
            }
            if(is_valid(nums1, nums2, mid) == -1) right = mid - 1;
            else left = mid + 1; 
        }
        int divider = left;

        if((len1 + len2) % 2 == 0) 
        {
            if(divider == 0)
            {
                if(res - divider == len2) return double(nums2[res - divider - 1] + nums1[divider]) / 2;
                else return double(nums2[res - divider - 1] + min(nums1[divider], nums2[res - divider])) / 2;
            }
            if(divider == len1)
            {
                if(res == divider) return double(nums1[divider - 1] + nums2[res - divider]) / 2;
                else return double((max(nums1[divider - 1], nums2[res - divider - 1]) + nums2[res - divider])) / 2;
            }
            return double((max(nums1[divider - 1], nums2[res - divider - 1]) + min(nums1[divider], nums2[res - divider]))) / 2;
        }
        else 
        {
            if(divider == 0) return nums2[res - divider - 1];
            if(divider == res) return nums1[divider - 1];
            return max(nums1[divider - 1], nums2[res - divider - 1]);
        }
    }

    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        return (nums1.size() <= nums2.size()) ? findDivider(nums1, nums2) : findDivider(nums2, nums1);
    }
};
*/
int main() {

    
    Solution solution;
    vector<int> nums3 = {4};
    vector<int> nums4 = {1,2,3,5};
    cout << "[1,3]和[2]的中位数: " 
         << solution.findMedianSortedArrays(nums3, nums4) << endl;
    
    return 0;
}

// 图论


/*
// T200 岛屿数量
// 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
// 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
// 此外，你可以假设该网格的四条边均被水包围。

class Solution {
public:
    int row, col;
    vector<vector<int>> directs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    vector<vector<int>> is_cal;
    bool is_val(int a, int b)
    {
        return (a >= 0) && (a < row) && (b >= 0) && (b < col);
    }

    int numIslands_bfs(vector<vector<char>>& grid) {
        row = grid.size(); col = grid[0].size();
        is_cal = vector<vector<int>>(row, vector<int>(col, 0)); // 是否被访问
        stack<vector<int>> s;
        int res = 0;
        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                if(is_cal[i][j] || (grid[i][j] == '0')) continue;
                res++;
                s.push({i, j});
                while(!s.empty())
                {
                    vector<int> t = s.top(); s.pop();
                    if(is_cal[t[0]][t[1]]) continue;
                    for(auto dir : directs)
                    {
                        // when point in the matrix && is not calculated && is land, it can be added into stack
                        if(is_val(t[0] + dir[0], t[1] + dir[1]) && (!is_cal[t[0] + dir[0]][t[1] + dir[1]]) && (grid[t[0] + dir[0]][t[1] + dir[1]] == '1'))
                        {
                            s.push({t[0] + dir[0], t[1] + dir[1]});
                        }
                    }
                    // record the t to be calculated
                    is_cal[t[0]][t[1]] = 1;
                }
            }
        }

        return res;
    }

    void dfs(vector<vector<char>>& grid, int a, int b)
    {
        is_cal[a][b] = 1; // already calculated
        for(auto dir : directs)
        {
            if(is_val(a + dir[0], b + dir[1]) && !is_cal[a + dir[0]][b + dir[1]] && grid[a + dir[0]][b + dir[1]] == '1') dfs(grid, a + dir[0], b + dir[1]);
        }
    }

    int numIslands_dfs(vector<vector<char>>& grid){
        row = grid.size(); col = grid[0].size();
        is_cal = vector<vector<int>>(row, vector<int>(col, 0)); // 是否被访问
        int res = 0;
        for(int i = 0;i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                if(is_cal[i][j] || (grid[i][j] == '0')) continue;
                dfs(grid, i, j);
                res++;
            }
        }
        return res;
    }
};
*/


/*
// T994 腐烂的橘子
// 在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
// 值 0 代表空单元格；
// 值 1 代表新鲜橘子；
// 值 2 代表腐烂的橘子。
// 每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。
// 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。

class Solution {
public:
    int row, col; // 长宽
    bool is_val(int a, int b)
    {
        return (a >= 0) && (a < row) && (b >= 0) && (b < col);
    }
    vector<vector<int>> directs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    
    int orangesRotting(vector<vector<int>>& grid) {
        row = grid.size(); col = grid[0].size();
        queue<vector<int>> s;
        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                if(grid[i][j] != 2) continue;
                s.push({i, j});
                grid[i][j] = -1; // 已经访问过
            }
        }

        int res = -1;
        while(!s.empty())
        {
            int l = s.size();
            for(int k = 0; k < l; k++)
            {
                vector<int> t = s.front(); s.pop();
                int x = t[0], y = t[1];
                for(auto d : directs)
                {
                    if(is_val(x + d[0], y + d[1]) && (grid[x + d[0]][y + d[1]] == 1)) {
                        s.push({x + d[0], y + d[1]});
                        grid[x + d[0]][y + d[1]] = -1; // 防止被这个循环的其他坏橘子影响
                    }
                }
            }
            res++;
        }
        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                if(grid[i][j] == 1) return -1;
            }
        }
        return (res == -1) ? 0 : res;
    }
};
*/


/*
// T207 课程表
// 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
// 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
// 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
// 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

// 基本算法：拓扑排序：找出所有入度为0的课程，将其加入表中，再减去它们指向的点的入度
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> l; // 储存顺序
        queue<int> q; // 度为0的节点
        vector<int> indegree(numCourses); // id[i]表示i的入度为多少
        vector<vector<int>> outdegree(numCourses); // od[i]表示i有哪些出度

        for(auto pr : prerequisites) // 初始化出度入度
        {
            indegree[pr[0]]++;
            outdegree[pr[1]].push_back(pr[0]);
        }
        for(int i = 0; i < numCourses; i++)
        {
            // 说明i课程是入度为0的课程
            if(indegree[i] == 0) q.push(i);
        }

        while(!q.empty())
        {
            int tmp = q.front(); q.pop(); // 学的课程
            l.push_back(tmp); 
            // 去掉tmp有关的出度和入度
            for(auto course : outdegree[tmp])
            {
                if(--indegree[course] == 0) q.push(course);
            }
        }

        return l.size() == numCourses;
    }
};
*/


/*
// T208 实现前缀树
// Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的
//应用情景，例如自动补全和拼写检查。

// 请你实现 Trie 类：

// Trie() 初始化前缀树对象。
// void insert(String word) 向前缀树中插入字符串 word 。
// boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
// boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
class Trie {
private:
    vector<Trie*> childs;
    bool is_end; // 判断是否有单词在这里终止
public:
    Trie() : childs(26, nullptr), is_end(false) {} // 表示a-z的摆放位置，使用成员初始化列表
    
    void insert(string word) {
        Trie* tmp = this; // 用于遍历
        for(int i = 0; i < word.size(); i++)
        {
            if(!tmp->childs[word[i] - 'a']) // 没有这个字母，需要分配一个空间
            {
                tmp->childs[word[i] - 'a'] = new Trie();
            }
            tmp = tmp->childs[word[i] - 'a'];
        }
        tmp->is_end = true; // 表示有单词在这里结束
    }
    
    bool search(string word) {
        Trie* tmp = this; // 用于遍历
        for(int i = 0; i < word.size(); i++)
        {
            if(!tmp->childs[word[i] - 'a']) return false;
            tmp = tmp->childs[word[i] - 'a'];
        }
        return tmp->is_end;
    }
    
    bool startsWith(string prefix) {
        Trie* tmp = this; // 用于遍历
        for(int i = 0; i < prefix.size(); i++)
        {
            if(!tmp->childs[prefix[i] - 'a']) return false; // 字母prefix[i]不存在，返回false
            tmp = tmp->childs[prefix[i] - 'a'];
        }
        return true;
    }
};
*/


