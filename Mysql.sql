-- T175 组合两个表
-- 编写解决方案，报告 Person 表中每个人的姓、名、城市和州。如果 personId 的地址不在 Address 表中，则报告为 null 。
-- Person表:
-- +----------+----------+-----------+
-- | personId | lastName | firstName |
-- +----------+----------+-----------+
-- | 1        | Wang     | Allen     |
-- | 2        | Alice    | Bob       |
-- +----------+----------+-----------+
-- Address表:
-- +-----------+----------+---------------+------------+
-- | addressId | personId | city          | state      |
-- +-----------+----------+---------------+------------+
-- | 1         | 2        | New York City | New York   |
-- | 2         | 3        | Leetcode      | California |
-- 输出: 
-- +-----------+----------+---------------+----------+
-- | firstName | lastName | city          | state    |
-- +-----------+----------+---------------+----------+
-- | Allen     | Wang     | Null          | Null     |
-- | Bob       | Alice    | New York City | New York |
-- +-----------+----------+---------------+----------+

-- 注：使用 outer join

-- SELECT p.firstName, p.lastName, a.city, a.state
-- FROM Person p LEFT JOIN Address a
-- on p.personId = a.personId;



-- T176 查询第二高的薪水
-- 查询并返回 Employee 表中第二高的 不同 薪水 。如果不存在第二高的薪水，查询应该返回 null(Pandas 则返回 None) 。
-- 输入：
-- Employee 表：
-- +----+--------+
-- | id | salary |
-- +----+--------+
-- | 1  | 100    |
-- | 2  | 200    |
-- | 3  | 300    |
-- +----+--------+
-- 输出：
-- +---------------------+
-- | SecondHighestSalary |
-- +---------------------+
-- | 200                 |
-- +---------------------+

-- Tips: Use `LIMIT` and `OFFSET`

-- 解法一
-- SELECT
--     (SELECT DISTINCT salary as s
--     FROM Employee ORDER BY salary DESC
--     LIMIT 1 OFFSET 1)
-- AS SecondHighestSalary;

-- 解法二
-- SELECT MAX(salary) AS SecondHighestSalary
-- FROM Employee
-- WHERE salary < (SELECT MAX(salary) FROM Employee);


-- T177 第n高的薪水
-- 查询 Employee 表中第 n 高的工资。如果没有第 n 个最高工资，查询结果应该为 null 。
-- 输入: 
-- Employee table:
-- +----+--------+
-- | id | salary |
-- +----+--------+
-- | 1  | 100    |
-- | 2  | 200    |
-- | 3  | 300    |
-- +----+--------+
-- n = 2
-- 输出: 
-- +------------------------+
-- | getNthHighestSalary |
-- +------------------------+
-- | 200                    |
-- +------------------------+
-- SELECT
--     (SELECT DISTINCT salary FROM Employee
--     ORDER BY salary DESC 
--     LIMIT 1 OFFSET N - 1)
-- AS getNthHighestSalary;


-- T178 分数排名
-- 编写一个解决方案来查询分数的排名。排名按以下规则计算:
-- 1. 分数应按从高到低排列。
-- 2. 如果两个分数相等，那么两个分数的排名应该相同。
-- 3. 在排名相同的分数后，排名数应该是下一个连续的整数。换句话说，排名之间不应该有空缺的数字。
-- 4. 按 score 降序返回结果表。                                                                                                                                
-- 输入: 
-- Scores 表:
-- +----+-------+
-- | id | score |
-- +----+-------+
-- | 1  | 3.50  |
-- | 2  | 3.65  |
-- | 3  | 4.00  |
-- | 4  | 3.85  |
-- | 5  | 4.00  |
-- | 6  | 3.65  |
-- +----+-------+
-- 输出: 
-- +-------+------+
-- | score | rank |
-- +-------+------+
-- | 4.00  | 1    |
-- | 4.00  | 1    |
-- | 3.85  | 2    |
-- | 3.65  | 3    |
-- | 3.65  | 3    |
-- | 3.50  | 4    |
-- +-------+------+
-- SELECT s1.score, 
-- (SELECT COUNT(DISTINCT s2.score)
--     FROM Scores s2
--     WHERE s2.score >= s1.score)
-- AS "rank"
-- FROM Scores s1
-- ORDER BY s1.score DESC;

