SELECT 
    Zips.Zip, 
    Zips.City, 
    Zips.Lat, 
    Zips.Lng, 
    COUNT(Orders.OrderId) AS total_orders 
FROM Zips
JOIN Customer ON Zips.Zip = Customer.PostalCode -- Join on PostalCode
JOIN Orders ON Customer.CustomerId = Orders.CustomerId -- Join on CustomerId
WHERE ST_DISTANCE_SPHERE(POINT(Zips.Lng, Zips.Lat), POINT(-71.104444, 42.364506)) / 1000 <= 100 -- Filter by distance
GROUP BY Zips.Zip, Zips.City, Zips.Lat, Zips.Lng -- Group by zip code and additional fields 
ORDER BY total_orders DESC -- Order by total orders in descending order
LIMIT 3; -- Limit to top 3 zip codes